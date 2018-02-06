import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import base, prepare_spec
from .. import beam_search
from datasets import data
from attention import SimpleSDPAttention


def default(value, if_none):
    return if_none if value is None else value


def expand(v, k):
    # Input: d1 x ...
    # Output: d1 * k x ... where
    #   out[0] = out[1] = ... out[k],
    #   out[k + 0] = out[k + 1] = ... out[k + k],
    # and so on.
    return v.unsqueeze(1).repeat(1, k, *([1] *
                                         (v.dim() - 1))).view(-1, *v.shape[1:])


def flatten(v, k):
    # Input: d1 x ... x dk x dk+1 x ... x dn
    # Output: d1 x ... x dk * dk+1 x ... x dn
    args = v.shape[:k] + (-1, ) + v.shape[k + 2:]
    return v.view(*args)


def maybe_concat(items, dim=None):
    to_concat = [item for item in items if item is not None]
    if not to_concat:
        return None
    elif len(to_concat) == 1:
        return to_concat[0]
    else:
        return torch.cat(to_concat, dim)


def take(tensor, indices):
    '''Equivalent of numpy.take for Torch tensors.'''
    indices_flat = indices.contiguous().view(-1)
    return tensor[indices_flat].view(indices.shape + tensor.shape[1:])


def lstm_init(cuda, num_layers, hidden_size, *batch_sizes):
    init_size = (num_layers, ) + batch_sizes + (hidden_size, )
    init = Variable(torch.zeros(*init_size))
    if cuda:
        init = init.cuda()
    return (init, init)


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
        batch_dims = input_grid.shape[:-3]
        input_grid = input_grid.contiguous().view(-1, 15, 18, 18)
        output_grid  = output_grid.contiguous().view(-1, 15, 18, 18)

        input_enc = self.input_encoder(input_grid)
        output_enc = self.output_encoder(output_grid)
        enc = torch.cat([input_enc, output_enc], 1)
        enc = enc + self.block_1(enc)
        enc = enc + self.block_2(enc)

        enc = self.fc(enc.view(*(batch_dims + (-1,))))
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
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def expand_by_beam(self, beam_size):
        v = self.value
        return LGRLMemory(expand(v, beam_size))


class LGRLRefineDecoderState(
        collections.namedtuple('LGRLRefineDecoderState',
                               ['context', 'h', 'c']),
        beam_search.BeamSearchState):
    # context: batch (* beam) x num pairs x hidden size
    # h: 2 x batch (* beam) x num pairs x hidden size
    # c: 2 x batch (* beam) x num pairs x hidden size

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        selected = [
            None if self.context is None else self.context.view(
                batch_size, -1, *self.context.shape[1:])[indices.data.numpy()]
        ]
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = v.view(2, batch_size, -1, *v.shape[2:])
            # result: 2 x indices.shape[1] x num pairs x hidden
            selected.append(v[(slice(None), ) + tuple(indices.data.numpy())])

        return LGRLRefineDecoderState(*selected)


class LGRLRefineMemory(beam_search.BeamSearchMemory):
    __slots__ = ('io', 'code', 'trace')

    def __init__(self, io, code, trace):
        # io: batch (* beam size) x num pairs x hidden size
        self.io = io
        # code: batch (* beam size) x code length x hidden size, or None
        self.code = code
        # trace: batch (* beam size) x num pairs x trace length x hidden size,
        # or None
        self.trace = trace

    def expand_by_beam(self, beam_size):
        io_exp = expand(self.io, beam_size)
        code_exp = None if self.code is None else self.code.expand_by_beam(
            beam_size)
        trace_exp = None if self.trace is None else self.trace.expand_by_beam(
            beam_size)
        return LGRLRefineMemory(io_exp, code_exp, trace_exp)


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
            input_size=256 + 512,
            hidden_size=256,
            num_layers=2,
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
        out_embed = expand(out_embed, pairs_per_example)

        # io_embed_exp shape: batch size * num pairs x length x hidden size
        io_embed_exp = io_embed.view(
            -1, io_embed.shape[-1]).unsqueeze(1).expand(-1, out_embed.shape[1],
                                                        -1)

        decoder_input = torch.cat([out_embed, io_embed_exp], dim=2)
        # decoder_output shape: batch size * num pairs x length x hidden size
        decoder_output, _ = self.decoder(
            decoder_input, self.zero_state(decoder_input.shape[0]))

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
        token_embed = token_embed.unsqueeze(1).expand(-1, io_embed.shape[1],
                                                      -1)
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

    def zero_state(self, *args):
        return lstm_init(self._cuda, 2, 256, *args)


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, args):
        super(CodeEncoder, self).__init__()

        self._cuda = args.cuda
        self.embed = nn.Embedding(vocab_size, 256)
        self.encoder = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True)

    def forward(self, inputs):
        # inputs: PackedSequencePlus, batch size x sequence length
        inp_embed = inputs.apply(self.embed)
        # output: PackedSequence, batch size x seq length x hidden (256 * 2)
        # state: 2 (layers) * 2 (directions) x batch x hidden size (256)
        output, state = self.encoder(inp_embed.ps,
                                     lstm_init(self._cuda, 4, 256,
                                               inp_embed.ps.batch_sizes[0]))

        return inp_embed.with_new_ps(output)


class TraceEncoder(nn.Module):
    def __init__(self, interleave_events, include_flow_events,
                 event_emb_from_code_seq):
        super(TraceEncoder, self).__init__()

        assert not include_flow_events or interleave_events
        assert not event_emb_from_code_seq or interleave_events

        self.interleave_events = interleave_events
        self.include_flow_events = include_flow_events
        self.event_emb_from_code_seq = event_emb_from_code_seq

    def forward(self, code_enc, traces):
        # code_enc: PackedSequencePlus, batch x seq x hidden size
        # traces: tuple of
        # - grids: PackedSequencePlus, batch x seq x 15 (channels) x 18 x 18
        # - events: list of list of tuple
        #   - timestep
        #   - type
        #     - actions: move, turnLeft/Right, put/pickMarker
        #     - control flow: if/ifelse/repeat
        #   - conditional:
        #      - front/left/rightIsClear, markerPresent, and inverse
        #      - R=1 to R=19?
        #   - conditional value: true, false, current loop iteration
        #   - code bounds: (left, right) for whole of action/control flow
        #
        # Returns: PackedSequencePlus, batch x seq x 512
        raise NotImplementedError


class TimeConvTraceEncoder(TraceEncoder):
    def __init__(
            self,
            args,
            time=3,
            channels=64,
            out_dim=512,
            # Whether to include actions/control flow in the middle
            interleave_events=False,
            # Also include control flow events in addition to actions
            include_flow_events=False,
            # Get embeddings from code_seq using boundary information
            event_emb_from_code_seq=False, ):
        super(TimeConvTraceEncoder, self).__init__(
            interleave_events, include_flow_events, event_emb_from_code_seq)
        self._cuda = args.cuda

        self.max_iterations = 9
        self.event_emb_dim = 32

        assert time % 2 == 1
        c = channels
        k = (time, 3, 3)
        p = ((time - 1) / 2, 1, 1)

        # Feature maps:
        # 15 for current state
        # interleave_events:
        #   +5 indicator for each of the 5 actions
        #   include_flow_events:
        #      +2 for if/ifelse
        #      +8 for front/left/rightIsClear, markerPresent and inverses
        #      +2 for whether condition evaluated to true/false
        #
        #      +1 for repeat
        #      +k-1 for R=2, ..., R=k max iterations
        #      +k+1 for 0, ..., k iterations remaining
        #   event_emb_from_code_seq:
        #      +d for embedding?
        in_channels = 15
        if self.interleave_events:
            in_channels += 5
            if include_flow_events:
                in_channels += 13 + 2 * self.max_iterations
            if event_emb_from_code_seq:
                in_channels += self.event_emb_dim
                self.event_emb_project = nn.Linear(512, self.event_emb_dim)

        self.initial_conv = nn.Conv3d(
            in_channels=in_channels, out_channels=c, kernel_size=k, padding=p)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm3d(c),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=c, out_channels=c, kernel_size=k, padding=p),
                nn.BatchNorm3d(c),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=c, out_channels=c, kernel_size=k, padding=p))
            for _ in range(3)
        ])

        self.fc = nn.Linear(c * 18 * 18, out_dim)

    def forward(self, code_enc, traces_grids, traces_events):
        if self.interleave_events:
            raise NotImplementedError

        # grids: batch x seq length x 15 (channels) x 18 x 18
        # TODO: Don't unsort the batch here so that we can call
        # pack_padded_sequence more easily later.
        grids, seq_lengths = traces_grids.pad(batch_first=True)
        #grids, seq_lengths = nn.utils.rnn.pad_packed_sequence(
        #    traces.grids, batch_first=True)
        # grids: batch x 15 x seq length x 18 x 18
        grids = grids.transpose(1, 2)

        enc = self.initial_conv(grids)
        for block in self.blocks:
            enc = enc + block(enc)
        # before: batch x c x seq length x 18 x 18
        # after:  batch x seq length x c x 18 x 18
        enc = enc.transpose(1, 2)

        # enc: batch x seq length x out_dim
        enc = self.fc(enc.contiguous().view(enc.shape[0], enc.shape[1], -1))

        return traces_grids.with_new_ps(
            nn.utils.rnn.pack_padded_sequence(
                enc, seq_lengths, batch_first=True))


class RecurrentTraceEncoder(TraceEncoder):
    def __init__(
            self,
            args,
            # Include initial/final states
            concat_io=False,
            # Whether to include actions/control flow in the middle
            interleave_events=True,
            # Only include actions or also include control flow events
            include_flow_events=False,
            # Get embeddings from code_seq using boundary information
            event_emb_from_code_seq=False, ):
        super(RecurrentTraceEncoder, self).__init__(
            interleave_events, include_flow_events, event_emb_from_code_seq)
        self._cuda = args.cuda

        self.encoder = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True)

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

        self.success_emb = nn.Embedding(2, 256)
        self.action_code_proj = nn.Linear(512, 256)
        self.cond_code_proj = nn.Linear(512, 256 / 4)
        # 2: false, true
        # 11: 0..10
        self.cond_emb = nn.Embedding(13, 256)

    def forward(self, code_enc, traces_grids, traces_events, cag_interleave):
        def net(inp):
            enc = self.initial_conv(inp)
            for block in self.blocks:
                enc = enc + block(enc)
            enc = self.grid_fc(enc.view(enc.shape[0], -1))
            return enc
        grid_embs = traces_grids.apply(net)

        if self.interleave_events:
            action_embs = traces_events.actions.apply(
                lambda d: self.action_code_proj(
                    code_enc.ps.data[traces_events.action_code_indices]) * self.success_emb(d[:, 1])
            )

            assert (traces_events.cond_code_indices >=
                    len(code_enc.ps.data)).data.sum() == 0
            cond_embs = traces_events.conds.apply(
                    lambda d:
                       # Shape: sum(cond trace lengths) x 256 after view
                        self.cond_code_proj(
                          # Shape: sum(cond trace lengths) x 4 x 512
                          take(code_enc.ps.data,
                              traces_events.cond_code_indices)).view(-1, 256)
                          * self.cond_emb(d[:, 4])
                          * self.success_emb(d[:, 5]))
            seq_embs = prepare_spec.interleave_packed_sequences((cond_embs,
                action_embs, grid_embs), cag_interleave)
        else:
            seq_embs = grid_embs

        # output: PackedSequence, batch size x seq length x hidden (256 * 2)
        # state: 2 (layers) * 2 (directions) x batch x hidden size (256)
        output, state = self.encoder(seq_embs.ps,
                                     lstm_init(self._cuda, 4, 256,
                                               seq_embs.ps.batch_sizes[0]))
        return seq_embs.with_new_ps(output)


class LGRLSeqRefineDecoder(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLSeqRefineDecoder, self).__init__()
        self.args = args
        self.has_memory = (self.args.karel_code_enc != 'none' or
                      self.args.karel_trace_enc != 'none')

        self.num_placeholders = args.num_placeholders
        self._cuda = args.cuda
        self.embed = nn.Embedding(vocab_size + self.num_placeholders, 256)
        self.decoder = nn.LSTM(
            input_size=256 + 512 + (512 if self.has_memory else 0),
            hidden_size=256,
            num_layers=2)

        self.code_attention = None
        self.trace_attention = None
        if self.args.karel_code_enc != 'none':
            self.code_attention = SimpleSDPAttention(256, 256 * 2)
        if self.args.karel_trace_enc != 'none':
            self.trace_attention = SimpleSDPAttention(256, 256 * 2)
        if (self.args.karel_code_enc == 'none' or
                self.args.karel_trace_enc == 'none'):
            self.context_proj = lambda x: x
        else:
            self.context_proj = nn.Linear(512 + 512, 512)
        self.out = nn.Linear(
            256 + (512 if self.has_memory else 0),
            vocab_size + self.num_placeholders,
            bias=False)

    def prepare_memory(self, io_embed, code_memory, trace_memory):
        # code_memory:
        #   PackedSequencePlus, batch size x code length x 512
        #   or None
        # trace_memory:
        #   PackedSequencePlus, batch size * num pairs x seq x 512
        #   or None
        batch_size = io_embed.shape[0]
        pairs_per_example = io_embed.shape[1]

        if code_memory is not None:
            # batch x code length x 512
            code_memory, code_lengths = code_memory.pad(batch_first=True)
            code_mask = base.get_attn_mask(code_lengths, self._cuda)
            code_memory = code_memory.unsqueeze(1).repeat(1, pairs_per_example,
                                                          1, 1)
            code_mask = code_mask.unsqueeze(1).repeat(1, pairs_per_example, 1)
            code_memory = base.MaskedMemory(code_memory, code_mask)

        if trace_memory is not None:
            # batch * num pairs x trace length x 512
            trace_memory, trace_lengths = trace_memory.pad(batch_first=True)
            trace_mask = base.get_attn_mask(trace_lengths, self._cuda)
            # batch x num pairs x trace length x 512
            trace_memory = trace_memory.view(batch_size, pairs_per_example,
                                             *trace_memory.shape[1:])
            trace_mask = trace_mask.view(batch_size, pairs_per_example,
                                         *trace_mask.shape[1:])
            trace_memory = base.MaskedMemory(trace_memory, trace_mask)

        return LGRLRefineMemory(io_embed, code_memory, trace_memory)

    def forward(self, io_embed, code_memory, trace_memory, outputs):
        # io_embed: batch size x num pairs x 512
        # code_memory:
        #   PackedSequencePlus, batch size x code length x 512
        #   or None
        # trace_memory:
        #   PackedSequencePlus, batch size * num pairs x seq x 512
        #   or None
        # outputs:
        #   batch size x output length
        batch_size = io_embed.shape[0]
        pairs_per_example = io_embed.shape[1]

        # Remove </S> from longest sequence
        # outputs shape: batch x length
        # labels shape: batch x length
        outputs, labels = outputs[:, :-1], outputs[:, 1:]

        # batch x length x hidden size
        out_embed = self.embed(data.replace_pad_with_end(outputs))
        # batch x num pairs x length x hidden size
        out_embed = out_embed.unsqueeze(1).expand(-1, pairs_per_example, -1,
                                                  -1)

        memory = self.prepare_memory(io_embed, code_memory, trace_memory)
        state = self.zero_state(out_embed.shape[0], out_embed.shape[1])

        all_logits = []
        for t in range(outputs.shape[1]):
            # batch x num pairs x hidden size
            out_emb = out_embed[:, :, t]
            state, logits = self.compute_next_token_logits(state, memory,
                                                           out_emb)
            all_logits.append(logits)

        all_logits = torch.stack(all_logits, dim=1)
        return all_logits, labels

    def decode_token(self, token, state, memory, attentions):
        pairs_per_example = memory.io.shape[1]

        # token: LongTensor, batch (* beam)
        token_emb = self.embed(token)
        # TODO handle attentions arg
        return self.compute_next_token_logits(
            state, memory,
            token_emb.unsqueeze(1).expand(-1, pairs_per_example, -1))

    def compute_next_token_logits(self, state, memory, last_token_emb):
        # state: LGRLRefineDecoderState
        #   context: batch (* beam) x num pairs x hidden size
        #   h: 2 x batch (* beam) x num pairs x hidden size
        #   c: 2 x batch (* beam) x num pairs x hidden size
        # memory: LGRLRefineMemory
        #   io: batch (* beam) x num pairs x hidden size
        #   code: batch (* beam) x num pairs (rep.) x code length x hidden size
        #   trace: batch (* beam) x num pairs x trace length x hidden size
        # last_token_emb: batch (* beam) x num pairs x hidden size
        pairs_per_example = memory.io.shape[1]

        decoder_input = torch.cat([last_token_emb, memory.io] + ([state.context]
                if self.has_memory else []), dim=2)
        decoder_input = decoder_input.view(-1, decoder_input.shape[-1])

        # decoder_output: 1 x batch (* beam) * num pairs x hidden size
        # new_state: length-2 tuple of
        #   2 x batch (* beam) * num pairs x hidden size
        decoder_output, new_state = self.decoder(
            # 1 x batch (* beam) * num pairs x hidden size
            decoder_input.unsqueeze(0),
            # v before: 2 x batch (* beam) x num pairs x hidden
            # v after:  2 x batch (* beam) * num pairs x hidden
            (flatten(state.h, 1), flatten(state.c, 1)))
        new_state = (new_state[0].view_as(state.h),
                     new_state[1].view_as(state.c))
        decoder_output = decoder_output.squeeze(0)

        code_context, trace_context = None, None
        if memory.code:
            code_context, _ = self.code_attention(
                decoder_output,
                flatten(memory.code.memory, 0),
                flatten(memory.code.attn_mask, 0))
        if memory.trace:
            trace_context, _ = self.trace_attention(
                decoder_output,
                flatten(memory.trace.memory, 0),
                flatten(memory.trace.attn_mask, 0))
        # batch (* beam) * num_pairs x hidden
        concat_context = maybe_concat([code_context, trace_context], dim=1)
        if concat_context is None:
            new_context = None
        else:
            new_context = self.context_proj(concat_context)

        # batch (* beam) * num pairs x hidden
        emb_for_logits = maybe_concat([new_context, decoder_output], dim=1)
        # batch (* beam) x hidden
        emb_for_logits, _ = emb_for_logits.view(
            -1, pairs_per_example, emb_for_logits.shape[-1]).max(dim=1)
        # batch (* beam) x vocab size
        logits = self.out(emb_for_logits)

        return LGRLRefineDecoderState(
            None if new_context is None else
            new_context.view(-1, pairs_per_example, new_context.shape[-1]),
            *new_state), logits

    def zero_state(self, *args):
        context_size = args + (512, )
        context = Variable(torch.zeros(*context_size))
        if self._cuda:
            context = context.cuda()
        return LGRLRefineDecoderState(context,
                                      *lstm_init(self._cuda, 2, 256, *args))


class LGRLKarel(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLKarel, self).__init__()
        self.args = args

        self.encoder = LGRLTaskEncoder(args)
        self.decoder = LGRLSeqDecoder(vocab_size, args)

    def encode(self, input_grid, output_grid):
        return self.encoder(input_grid, output_grid)

    def decode(self, io_embed, outputs):
        return self.decoder(io_embed, outputs)

    def decode_token(self, token, state, memory, attentions=None):
        return self.decoder.decode_token(token, state, memory, attentions)


class LGRLRefineKarel(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLRefineKarel, self).__init__()
        self.args = args

        if self.args.karel_trace_enc == 'conv3d':
            self.trace_encoder = TimeConvTraceEncoder(self.args)
        elif self.args.karel_trace_enc == 'lstm':
            self.trace_encoder = RecurrentTraceEncoder(self.args)
        elif self.args.karel_trace_enc == 'none':
            self.trace_encoder = lambda *args: None
        else:
            raise ValueError(self.args.karel_trace_enc)

        if self.args.karel_code_enc == 'default':
            self.code_encoder = CodeEncoder(vocab_size, args)
        elif self.args.karel_code_enc == 'none':
            self.code_encoder = lambda *args: None
        else:
            raise ValueError(self.args.karel_code_enc)

        self.encoder = LGRLTaskEncoder(args)
        self.decoder = LGRLSeqRefineDecoder(vocab_size, args)

    def encode(self, input_grid, output_grid, ref_code, ref_trace_grids,
               ref_trace_events, cag_interleave):
        # batch size x num pairs x 512
        io_embed = self.encoder(input_grid, output_grid)
        # PackedSequencePlus, batch size x length x 512
        ref_code_memory = self.code_encoder(ref_code)
        # PackedSequencePlus, batch size x num pairs x length x  512
        ref_trace_memory = self.trace_encoder(ref_code_memory, ref_trace_grids,
                                              ref_trace_events, cag_interleave)
        return io_embed, ref_code_memory, ref_trace_memory

    def decode(self, io_embed, ref_code_memory, ref_trace_memory, outputs):
        return self.decoder(io_embed, ref_code_memory, ref_trace_memory,
                            outputs)

    def decode_token(self, token, state, memory, attentions=None):
        return self.decoder.decode_token(token, state, memory, attentions)
