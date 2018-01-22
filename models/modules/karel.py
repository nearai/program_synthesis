import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .. import beam_search
from datasets import data


class LGRLTaskEncoder(nn.Module):
    '''Implements the encoder from:

    Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis
    https://openreview.net/forum?id=H1Xw62kRZ
    '''

    def __init__(self, args):
        super(LGRLTaskEncoder, self).__init__()

        self.input_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=15, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(), )
        self.output_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=15, out_channels=32, kernel_size=3, padding=1),
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

        self.fc = nn.Linear(64 * 18 * 18, 512)

    def forward(self, input_grid, output_grid):
        input_enc = self.input_encoder(input_grid)
        output_enc = self.output_encoder(output_grid)
        enc = torch.cat([input_enc, output_enc], 1)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)
        enc = self.fc(enc.view(enc.shape[0], -1))
        return enc


class LGRLDecoderState(
        collections.namedtuple('LGRLDecoderState', ['h', 'c']),
        beam_search.BeamSearchState):
    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        selected = []
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = v.view(2, batch_size, -1, *v.shape[2:])
            # result: 2 x indices.shape[1] x num pairs x hidden
            selected.append(v[(slice(None), ) + tuple(indices.data.numpy())])
        return LGRLDecoderState(*selected)


class LGRLMemory(beam_search.BeamSearchMemory):
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def expand_by_beam(self, beam_size):
        v = self.value
        return LGRLMemory(
            v.unsqueeze(1).repeat(1, beam_size, *([1] * (v.dim() - 1))).view(
                -1, *v.shape[1:]))


class LGRLSeqDecoder(nn.Module):
    '''Implements the decoder from:

    Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis
    https://openreview.net/forum?id=H1Xw62kRZ
    '''

    def __init__(self, vocab_size, args):
        super(LGRLSeqDecoder, self).__init__()

        self.num_placeholders = args.num_placeholders
        self._cuda = args.cuda
        self.embed = nn.Embedding(vocab_size + self.num_placeholders, 256)
        self.decoder = nn.LSTM(
            input_size=256 + 512, hidden_size=256, num_layers=2,
            batch_first=True)
        self.out = nn.Linear(
                256, vocab_size + self.num_placeholders, bias=False)

    def forward(self, io_embed, outputs):
        # io_embed shape: batch size x num pairs x hidden size
        pairs_per_example = io_embed.shape[1]
        # Remove </S> from longest sequence
        outputs, labels = outputs[:, :-1], outputs[:, 1:]
        # out_embed shape: batch x length x hidden size
        out_embed = self.embed(data.replace_pad_with_end(outputs))
        out_embed = out_embed.unsqueeze(1).repeat(
            1, pairs_per_example, 1, 1).view(-1, *out_embed.shape[1:])

        # io_embed_exp shape: batch size * num pairs x length x hidden size
        io_embed_exp = io_embed.view(-1, io_embed.shape[-1]).unsqueeze(1).expand(-1, out_embed.shape[1], -1)

        decoder_input = torch.cat([out_embed, io_embed_exp], dim=2)
        # decoder_output shape: batch size * num pairs x length x hidden size
        decoder_output, _ = self.decoder(decoder_input,
                self.zero_state(decoder_input.shape[0]))

        # decoder_output shape: batch size x length x hidden size
        decoder_output, _ = decoder_output.contiguous().view(
            -1, pairs_per_example, *decoder_output.shape[1:]).max(dim=1)

        logits = self.out(decoder_output)
        return logits, labels

    def decode_token(self, token, state, io_embed, attentions=None):
        # TODO: deduplicate logic with forward()

        # token: batch size (1D LongTensor)
        # state: LGRLDecoderState
        # io_embed: LGRLMemory, containing batch size (* beam size) x num pairs
        # x hidden size
        io_embed = io_embed.value
        pairs_per_example = io_embed.shape[1]

        # token_embed shape: batch size (* beam size) x hidden size
        token_embed = self.embed(token)
        # batch size (* beam size) x num pairs x hidden size
        token_embed = token_embed.unsqueeze(1).repeat(1, io_embed.shape[1], 1)
        # batch size (* beam size) x num pairs x hidden size
        decoder_input = torch.cat([token_embed, io_embed], dim=2)
        decoder_output, new_state = self.decoder(
            # batch size (* beam size) * num pairs x 1 x hidden size
            decoder_input.view(-1, decoder_input.shape[-1]).unsqueeze(1),
            # v before: 2 x batch size (* beam size) x num pairs x hidden
            # v after:  2 x batch size (* beam size) * num pairs x hidden
            tuple(v.view(v.shape[0], -1, v.shape[-1]) for v in state))
        new_state = LGRLDecoderState(*(v.view(v.shape[0], -1,
                                              pairs_per_example, v.shape[-1])
                                       for v in new_state))

        # shape after squeezing: batch size (* beam size) * num pairs x hidden
        decoder_output = decoder_output.squeeze(1)
        decoder_output = decoder_output.view(-1, pairs_per_example,
                                             *decoder_output.shape[1:])
        decoder_output, _ = decoder_output.max(dim=1)
        logits = self.out(decoder_output)

        return new_state, logits

    def zero_state(self, batch_size):
        init = Variable(torch.zeros(2, batch_size, 256))
        if self._cuda:
            init = init.cuda()
        return (init, init)


class LGRLKarel(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLKarel, self).__init__()
        self.args = args
        self._cuda = args.cuda

        self.encoder = LGRLTaskEncoder(args)
        self.decoder = LGRLSeqDecoder(vocab_size, args)

    def encode(self, input_grid, output_grid):
        return self.encoder(input_grid, output_grid)

    def decode(self, io_embed, outputs):
        return self.decoder(io_embed, outputs)

    def decode_token(self, token, state, unused_memory, attentions=None):
        return self.decoder.decode_token(token, state, unused_memory, attentions)
