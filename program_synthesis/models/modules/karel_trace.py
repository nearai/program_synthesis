import collections

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import karel as karel_modules
from .. import beam_search
from program_synthesis.datasets.karel import karel_runtime


action_to_id = {
    '<s>': 0,
    '</s>': 1,
    'move': 2,
    'turnLeft': 3,
    'turnRight': 4,
    'putMarker': 5,
    'pickMarker': 6,
}
id_to_action = {
    0: '<s>',
    1: '</s>',
    2: 'move',
    3: 'turnLeft',
    4: 'turnRight',
    5: 'putMarker',
    6: 'pickMarker',
}


class PResNetGridEncoder(nn.Module):

    def __init__(self, args):
        super(PResNetGridEncoder, self).__init__()

        # TODO: deduplicate with one in karel.py
        # (which will break checkpoints?)
        self.initial_conv = nn.Conv2d(
            in_channels=15, out_channels=64, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, padding=1))
            for _ in range(3)
        ])
        self.grid_fc = nn.Linear(64 * 18 * 18, 256)

    def forward(self, grids):
        # grids: batch size x 15 x 18 x 18
        enc = self.initial_conv(grids)
        for block in self.blocks:
            enc = enc + block(enc)
        enc = self.grid_fc(enc.view(enc.shape[0], -1))
        return enc


class LGRLGridEncoder(nn.Module):
    def __init__(self, args):
        super(LGRLGridEncoder, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=15, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), )

        self.grid_fc = nn.Linear(64 * 18 * 18, 256)

    def forward(self, grids):
        # grids: batch size x 15 x 18 x 18
        enc = self.initial_conv(grids)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)
        enc = self.grid_fc(enc.view(enc.shape[0], -1))
        return enc


class TraceDecoderState(
        collections.namedtuple( 'TraceDecoderState', ['field', 'h', 'c']),
        beam_search.BeamSearchState):
    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        # field: batch size (* beam size) x 15 x 18 x 18, numpy.ndarray
        selected = [
            self.field.reshape(
                batch_size, -1,
                *self.field.shape[1:])[tuple(indices.data.numpy())]
        ]
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = v.view(2, batch_size, -1, *v.shape[2:])
            # result: 2 x indices.shape[1] x num pairs x hidden
            selected.append(v[(slice(None), ) + tuple(indices.data.numpy())])
        return TraceDecoderState(*selected)


class TraceDecoder(nn.Module):
    def __init__(self, args):
        super(TraceDecoder, self).__init__()
        self._cuda = args.cuda

        if args.karel_trace_grid_enc == 'lgrl':
            self.grid_encoder = LGRLGridEncoder(args)
        elif args.karel_trace_grid_enc == 'presnet':
            self.grid_encoder = PResNetGridEncoder(args)
        elif args.karel_trace_grid_enc == 'none':
            self.grid_encoder = None
        else:
            raise ValueError(args.karel_trace_grid_enc)

        self.decoder = nn.LSTM(
            input_size=256 + (256 if self.grid_encoder else 0) +  512,
            hidden_size=256,
            num_layers=2)

        # Actions:
        # <s>, </s>, move, turn{Left,Right}, {pick,put}Marker
        num_actions = 7
        self.action_embed = nn.Embedding(num_actions, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self, io_embed, trace_grids, input_actions, output_actions,
            io_embed_indices):
        # io_embed: batch size x 512
        # trace_grids: PackedSequencePlus
        #   batch size x trace length x 15 x 18 x 18
        # input_actions: PackedSequencePlus
        #   batch size x trace length
        # output_actions: PackedSequencePlus
        #   batch size x trace length
        # io_embed_indices: len(input_actions)

        # PackedSequencePlus, batch size x sequence length x 256
        input_actions = input_actions.apply(self.action_embed)

        if self.grid_encoder:
            # PackedSequencePlus, batch size x sequence length x 256
            trace_grids = trace_grids.apply(self.grid_encoder)
            dec_input = input_actions.apply(
                    lambda d: torch.cat((d, trace_grids.ps.data,
                        io_embed[io_embed_indices]), dim=1))
        else:
            dec_input = input_actions.apply(
                lambda d: torch.cat((d, io_embed[io_embed_indices]), dim=1))

        dec_output, state = self.decoder(dec_input.ps)

        logits = self.out(dec_output.data)
        return logits, output_actions.ps.data

    def decode_token(self, token, state, memory, attentions=None):
        io_embed = memory.value

        # Advance the grids with the last action
        if self.grid_encoder:
            kr = karel_runtime.KarelRuntime()
            fields = state.field.copy()
            for field, action_id in zip(fields, token.data.cpu()):
                if action_id < 2:  # Ignore <s>, </s>
                    continue
                kr.init_from_array(field)
                getattr(kr, id_to_action[action_id])()
            fields_t =  Variable(torch.from_numpy(fields.astype(np.float32)))
            if self._cuda:
                fields_t = fields_t.cuda()
            grid_embed = self.grid_encoder(fields_t)
        else:
            fields = state.field
            grid_embed = None

        # action_embed shape: batch size (* beam size) x 256
        action_embed = self.action_embed(token)

        # batch size (* beam size) x (256 + 256 + 512)
        dec_input = karel_modules.maybe_concat(
                (action_embed, grid_embed, io_embed), dim=1)
        dec_output, new_state = self.decoder(
            # 1 x batch size (* beam size) x hidden size
            dec_input.unsqueeze(0),
            (state.h, state.c))

        # Shape after squeezing: batch size (* beam size) x 256
        dec_output = dec_output.squeeze(0)
        logits = self.out(dec_output)

        return TraceDecoderState(fields, *new_state), logits

    def init_state(self, *args):
        return karel_modules.lstm_init(self._cuda, 2, 256, *args)


class TracePrediction(nn.Module):
    def __init__(self, args):
        super(TracePrediction, self).__init__()

        self.encoder = karel_modules.LGRLTaskEncoder(args)
        self.decoder = TraceDecoder(args)

    def encode(self, input_grids, output_grids):
        # input_grids: batch size x 15 x 18 x 18
        # output_grids: batch size x 15 x 18 x 18
        return self.encoder(input_grids, output_grids)

    def decode(self, *args):
        return self.decoder(*args)

    def decode_token(self, token, state, memory, attentions=None):
        return self.decoder.decode_token(token, state, memory, attentions)
