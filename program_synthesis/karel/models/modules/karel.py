import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from program_synthesis.common.models import beam_search
from program_synthesis.common.modules.attention import SimpleSDPAttention

from program_synthesis.karel.dataset import data
from program_synthesis.karel.models import base
from program_synthesis.karel.models import prepare_spec
from program_synthesis.karel.models.modules import karel_common
from program_synthesis.karel.models.modules import utils


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
        return LGRLMemory(utils.expand(v, beam_size))


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
        io_exp = utils.expand(self.io, beam_size)
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
        out_embed = utils.expand(out_embed, pairs_per_example)

        # io_embed_exp shape: batch size * num pairs x length x hidden size
        io_embed_exp = io_embed.view(
            -1, io_embed.shape[-1]).unsqueeze(1).expand(-1, out_embed.shape[1],
                                                        -1)

        decoder_input = torch.cat([out_embed, io_embed_exp], dim=2)
        # decoder_output shape: batch size * num pairs x length x hidden size
        decoder_output, _ = self.decoder(
            decoder_input, self.init_state(decoder_input.shape[0]))

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

    def postprocess_output(self, sequences, _):
        return sequences

    def init_state(self, *args):
        return utils.lstm_init(self._cuda, 2, 256, *args)


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
                                     utils.lstm_init(self._cuda, 4, 256,
                                               inp_embed.ps.batch_sizes[0]))

        return utils.EncodedSequence(
                inp_embed.with_new_ps(output),
                state)


class CodeUpdater(nn.Module):
    def __init__(self,  args):
        super(CodeUpdater, self).__init__()
        self._cuda = args.cuda
        self.encoder = nn.LSTM(
                input_size=512 + 512,
                hidden_size=256,
                num_layers=1,
                bidirectional=True)

        #self.zero_memory = Variable(torch.zeros(1, 256))
        self.trace_gate = nn.Sequential(
                nn.Linear(512 + 512, 512),
                nn.Sigmoid())

    def forward(self, code_memory, trace_memory, code_update_info):
        # For each code position, find all of the times that the code has been
        # referenced in the execution trace: action and cond events.
        # - Scale each one by a sigmoid and then sum?
        # - Run a separate RNN over each one?
        #
        # For each x_i, find t_i1, ..., t_ik.
        # r_ik = sigmoid(W [x_i t_i1])
        # x_i' = concat(x_i, sum_k (r_ik * t_ik))
        # inputs: PackedSequencePlus, batch size x seq length x hidden (256 * 2)
        #
        # code_indices: indices of code tokens
        # trace_indices: indices of trace events

        selected_code_memory = code_memory.mem.ps.data[
            code_update_info.code_indices]
        selected_trace_memory = trace_memory.mem.ps.data[
            code_update_info.trace_indices]

        # Shape: num items x 512
        trace_gates = self.trace_gate(torch.cat(
                (selected_code_memory, selected_trace_memory), dim=1))
        gated_trace_memory = trace_gates * selected_trace_memory

        # max_trace_refs:
        #   maximum number of times that some token is referenced in the trace
        # code length x max trace refs x 256
        code_trace_update = Variable(
            torch.zeros(
                code_memory.mem.ps.data.shape[0] *
                code_update_info.max_trace_refs,
                512,
                out=torch.cuda.FloatTensor()
                if self._cuda else torch.FloatTensor()))

        code_trace_update.index_copy_(
            0, code_update_info.code_trace_update_indices, gated_trace_memory)
        # code length x 256
        code_trace_update = code_trace_update.view(
            code_memory.mem.ps.data.shape[0], code_update_info.max_trace_refs,
            512).sum(dim=1)

        updates, new_state = self.encoder(
                code_memory.mem.apply(
                    lambda t: torch.cat([t, code_trace_update], dim=1)).ps)

        code_memory = code_memory.mem.apply(lambda t: t + updates.data)
        return utils.EncodedSequence(code_memory, new_state)


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

        return utils.EncodedSequence(traces_grids.with_new_ps(
            nn.utils.rnn.pack_padded_sequence(
                enc, seq_lengths, batch_first=True)), None)


class RecurrentTraceEncoder(TraceEncoder):
    def __init__(
            self,
            args,
            # Include initial/final states
            concat_io=False,
            # Whether to include actions/control flow in the middle
            interleave_events=True,
            # Only include actions or also include control flow events
            include_flow_events=True,
            # Get embeddings from code_seq using boundary information
            event_emb_from_code_seq=True, ):
        super(RecurrentTraceEncoder, self).__init__(
            interleave_events, include_flow_events, event_emb_from_code_seq)
        self._cuda = args.cuda

        pieces = args.karel_trace_enc.split(':')
        if 'noev' in pieces:
            self.interleave_events = False
        if 'nocond' in pieces:
            self.include_flow_events = False
            raise NotImplementedError
        if 'noact' in pieces:
            raise NotImplementedError
        if 'staticemb' in pieces:
            raise NotImplementedError

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
                    code_enc.mem.ps.data[traces_events.action_code_indices]) * self.success_emb(d[:, 1])
            )

            assert (traces_events.cond_code_indices.data >=
                    len(code_enc.mem.ps.data)).sum() == 0
            cond_embs = traces_events.conds.apply(
                    lambda d:
                       # Shape: sum(cond trace lengths) x 256 after view
                        self.cond_code_proj(
                          # Shape: sum(cond trace lengths) x 4 x 512
                          utils.take(code_enc.mem.ps.data,
                              traces_events.cond_code_indices)).view(-1, 256)
                          * self.cond_emb(d[:, 4])
                          * self.success_emb(d[:, 5]))
            seq_embs = prepare_spec.execute_interleave_psps((cond_embs,
                action_embs, grid_embs), cag_interleave)
        else:
            seq_embs = grid_embs

        # output: PackedSequence, batch size x seq length x hidden (256 * 2)
        # state: 2 (layers) * 2 (directions) x batch x hidden size (256)
        output, state = self.encoder(seq_embs.ps,
                                     utils.lstm_init(self._cuda, 4, 256,
                                               seq_embs.ps.batch_sizes[0]))
        return utils.EncodedSequence(seq_embs.with_new_ps(output), state)


class LGRLSeqRefineDecoder(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLSeqRefineDecoder, self).__init__()
        self.args = args
        self.num_placeholders = args.num_placeholders
        self._cuda = args.cuda

        code_enc = self.args.karel_code_enc != 'none'
        trace_enc = self.args.karel_trace_enc != 'none'
        code_usage = set(x for x in args.karel_code_usage.split(',') if x)
        trace_usage = set(x for x in args.karel_trace_usage.split(',') if x)
        self.use_code_memory = utils.set_pop(code_usage, 'memory')
        self.use_code_state = utils.set_pop(code_usage, 'state')
        self.use_trace_memory = utils.set_pop(trace_usage, 'memory')
        self.use_trace_state = utils.set_pop(trace_usage, 'state')
        assert not code_usage
        assert not trace_usage

        self.has_memory = (code_enc and self.use_code_memory) or (
            trace_enc and self.use_trace_memory)

        self.embed = nn.Embedding(vocab_size + self.num_placeholders, 256)
        self.decoder = nn.LSTM(
            input_size=256 + 512 + (512 if self.has_memory else 0),
            hidden_size=256,
            num_layers=2)

        self.code_attention = None
        self.trace_attention = None
        if code_enc and self.use_code_memory:
            self.code_attention = SimpleSDPAttention(256, 256 * 2)
        if trace_enc and self.use_trace_memory:
            self.trace_attention = SimpleSDPAttention(256, 256 * 2)
        if self.code_attention and self.trace_attention:
            self.context_proj = nn.Linear(512 + 512, 512)
        else:
            self.context_proj = lambda x: x
        self.out = nn.Linear(
            256 + (512 if self.has_memory else 0),
            vocab_size + self.num_placeholders,
            bias=False)

        self.num_state_inputs = self.use_code_state + self.use_trace_state
        self.state_h_proj = None
        self.state_c_proj = None
        if self.num_state_inputs > 0:
            self.state_h_proj = torch.nn.ModuleList([
                    nn.Linear(512 * self.num_state_inputs, 256) for _ in
                    range(2)])
            self.state_c_proj = torch.nn.ModuleList([ nn.Linear(512 *
                self.num_state_inputs, 256) for _ in range(2)])

    def prepare_memory(self, io_embed, code_memory, trace_memory, _):
        # code_memory:
        #   EncodedSequence, containing:
        #     mem: PackedSequencePlus,
        #          batch size * num pairs x code length x 512
        #     state: tuple containing two of
        #       2 (layers) * 2 (directions) x batch size * num pairs x 256
        #   or None
        # trace_memory:
        #    Same as code_memory
        batch_size = io_embed.shape[0]
        pairs_per_example = io_embed.shape[1]

        if self.use_code_memory and code_memory is not None:
            # batch x code length x 512
            code_memory, code_lengths = code_memory.mem.pad(batch_first=True)
            code_mask = base.get_attn_mask(code_lengths, self._cuda)
            #code_memory = code_memory.unsqueeze(1).repeat(1, pairs_per_example,
            #                                              1, 1)
            #code_mask = code_mask.unsqueeze(1).repeat(1, pairs_per_example, 1)
            # batch x num pairs x code length x 512
            code_memory = code_memory.view(batch_size, pairs_per_example,
                                             *code_memory.shape[1:])
            code_mask = code_mask.view(batch_size, pairs_per_example,
                                         *code_mask.shape[1:])
            code_memory = utils.MaskedMemory(code_memory, code_mask)
        else:
            code_memory = None

        if self.use_trace_memory and trace_memory is not None:
            # batch * num pairs x trace length x 512
            trace_memory, trace_lengths = trace_memory.mem.pad(batch_first=True)
            trace_mask = base.get_attn_mask(trace_lengths, self._cuda)
            # batch x num pairs x trace length x 512
            trace_memory = trace_memory.view(batch_size, pairs_per_example,
                                             *trace_memory.shape[1:])
            trace_mask = trace_mask.view(batch_size, pairs_per_example,
                                         *trace_mask.shape[1:])
            trace_memory = utils.MaskedMemory(trace_memory, trace_mask)
        else:
            trace_memory = None

        return LGRLRefineMemory(io_embed, code_memory, trace_memory)

    def forward(self, io_embed, code_memory, trace_memory, outputs, _):
        # io_embed: batch size x num pairs x 512
        # code_memory:
        #   EncodedSequence, containing:
        #     mem: PackedSequencePlus,
        #          batch size (* num pairs) x code length x 512
        #     state: tuple containing two of
        #       2 (layers) * 2 (directions) x batch size (* num pairs) x 256
        #   or None
        # trace_memory:
        #   EncodedSequence, containing:
        #     mem: PackedSequencePlus,
        #          batch size * num pairs x code length x 512
        #     state: tuple containing two of
        #       2 (layers) * 2 (directions) x batch size * num pairs x 256
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

        memory = self.prepare_memory(
                io_embed, code_memory, trace_memory, None)
        state = self.init_state(code_memory, trace_memory,
                out_embed.shape[0], out_embed.shape[1])

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

        decoder_input = utils.maybe_concat([last_token_emb, memory.io, state.context
                if self.has_memory else None], dim=2)
        decoder_input = decoder_input.view(-1, decoder_input.shape[-1])

        # decoder_output: 1 x batch (* beam) * num pairs x hidden size
        # new_state: length-2 tuple of
        #   2 x batch (* beam) * num pairs x hidden size
        decoder_output, new_state = self.decoder(
            # 1 x batch (* beam) * num pairs x hidden size
            decoder_input.unsqueeze(0),
            # v before: 2 x batch (* beam) x num pairs x hidden
            # v after:  2 x batch (* beam) * num pairs x hidden
            (utils.flatten(state.h, 1), utils.flatten(state.c, 1)))
        new_state = (new_state[0].view_as(state.h),
                     new_state[1].view_as(state.c))
        decoder_output = decoder_output.squeeze(0)

        code_context, trace_context = None, None
        if memory.code:
            code_context, _ = self.code_attention(
                decoder_output,
                utils.flatten(memory.code.memory, 0),
                utils.flatten(memory.code.attn_mask, 0))
        if memory.trace:
            trace_context, _ = self.trace_attention(
                decoder_output,
                utils.flatten(memory.trace.memory, 0),
                utils.flatten(memory.trace.attn_mask, 0))
        # batch (* beam) * num_pairs x hidden
        concat_context = utils.maybe_concat([code_context, trace_context], dim=1)
        if concat_context is None:
            new_context = None
        else:
            new_context = self.context_proj(concat_context)

        # batch (* beam) * num pairs x hidden
        emb_for_logits = utils.maybe_concat([new_context, decoder_output], dim=1)
        # batch (* beam) x hidden
        emb_for_logits, _ = emb_for_logits.view(
            -1, pairs_per_example, emb_for_logits.shape[-1]).max(dim=1)
        # batch (* beam) x vocab size
        logits = self.out(emb_for_logits)

        return LGRLRefineDecoderState(
            None if new_context is None else
            new_context.view(-1, pairs_per_example, new_context.shape[-1]),
            *new_state), logits

    def postprocess_output(self, sequences, memory):
        return sequences

    def init_state(self, ref_code_memory, ref_trace_memory, batch_size,
                   pairs_per_example):
        context_size = (batch_size, pairs_per_example, 512)
        context = Variable(torch.zeros(*context_size))
        if self._cuda:
            context = context.cuda()

        if self.num_state_inputs > 0:
            # h, c: layers * directions x batch * num pairs x hidden
            h, c = [
                utils.maybe_concat(
                    [
                       utils.unexpand(ref_code_memory.state[i], pairs_per_example,
                           dim=1)
                        if self.use_code_state else None,
                        utils.unexpand(ref_trace_memory.state[i], pairs_per_example,
                            dim=1)
                        if self.use_trace_state else None
                    ],
                    dim=2) for i in (0, 1)
            ]
            new_state = []
            for s, proj in zip((h, c), (self.state_h_proj, self.state_c_proj)):
                # Shape: layers (2) x directions (2) x batch x num pairs x hidden size
                s = s.contiguous().view(-1, 2, *s.shape[1:])
                # Shape: layers (2) x batch x num pairs x directions (2) x hidden size
                s = s.permute(0, 2, 3, 1, 4)
                # Shape: layers (2) x batch x num pairs x directions * hidden size
                s = s.contiguous().view(*(s.shape[:3] + (-1,)))
                new_s = []
                for s_layer, proj_layer in zip(s, proj):
                    # Input: batch x num pairs x directions * hidden size (=512)
                    # Output: batch x num pairs x 256
                    new_s.append(proj_layer(s_layer))
                # Shape: 2 x batch x pairs per example x 256
                new_state.append(torch.stack(new_s))
            return LGRLRefineDecoderState(context,
                                          *new_state)

        return LGRLRefineDecoderState(context, *utils.lstm_init(
            self._cuda, 2, 256, batch_size, pairs_per_example))


class LGRLRefineEditDecoderState(
        collections.namedtuple('LGRLRefineEditDecoderState', [
            'source_locs', 'finished', 'context', 'h', 'c',
        ]),
        # source_locs: batch size (* beam size)
        beam_search.BeamSearchState):

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        indices_tuple = tuple(indices.data.numpy())
        selected = [
            self.source_locs.reshape(batch_size, -1)[indices_tuple],
            self.finished.reshape(batch_size, -1)[indices_tuple],
            None if self.context is None else self.context.view(
                batch_size, -1, *self.context.shape[1:])[indices_tuple]
        ]
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = v.view(2, batch_size, -1, *v.shape[2:])
            # result: 2 x indices.shape[1] x num pairs x hidden
            selected.append(v[(slice(None), ) + indices_tuple])
        return LGRLRefineEditDecoderState(*selected)

    def truncate(self, k):
        return LGRLRefineEditDecoderState(
                self.source_locs[:k],
                self.finished[:k],
                self.context[:k],
                self.h[:, :k],
                self.c[:, :k])


class LGRLRefineEditMemory(beam_search.BeamSearchMemory):
    __slots__ = ('io', 'code', 'ref_code')

    def __init__(self, io, code, ref_code):
        # io: batch (* beam size) x num pairs x hidden size
        self.io = io
        # code: PackedSequencePlus, batch (* beam size) x code length x 512
        self.code = code
        # ref_code: PackedSequencePlus, batch x seq length
        self.ref_code = ref_code

    def expand_by_beam(self, beam_size):
        io_exp = utils.expand(self.io, beam_size)
        if self.code is None:
            code_exp = None
        elif isinstance(self.code, utils.MaskedMemory):
            code_exp = self.code.expand_by_beam(beam_size)
        else:
            code_exp = self.code.expand(beam_size)
        ref_code_exp = self.ref_code.expand(beam_size)
        #source_tokens_exp = [
        #    tokens for tokens in self.source_tokens for _ in range(beam_size)
        #]
        #return LGRLRefineMemory(io_exp, code_exp, source_tokens_exp)
        return LGRLRefineEditMemory(io_exp, code_exp, ref_code_exp)


class LGRLSeqRefineEditDecoder(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLSeqRefineEditDecoder, self).__init__()
        assert args.karel_code_enc != 'none'
        assert args.num_placeholders == 0
        self.args = args
        self._cuda = args.cuda

        code_usage = set(args.karel_code_usage.split(','))
        self.use_code_memory = utils.set_pop(code_usage, 'memory')
        self.use_code_attn = utils.set_pop(code_usage, 'memory-attn')
        self.use_code_state = utils.set_pop(code_usage, 'state')
        assert not code_usage
        assert not (self.use_code_memory and self.use_code_attn)

        num_ops = 4 + 2 * vocab_size

        self.op_embed = nn.Embedding(num_ops, 256)
        self.last_token_embed = nn.Embedding(vocab_size, 256)
        self.decoder = nn.LSTM(
            input_size=512 + 512 +
            (512 if self.use_code_memory or self.use_code_attn else 0),
            hidden_size=256,
            num_layers=2)
        self.out = nn.Linear(
            256 + (512 if self.use_code_attn else 0), num_ops, bias=False)

        if self.use_code_attn:
            self.code_attention = SimpleSDPAttention(256, 256 * 2)

        # Given the past op, whether to increment source_loc or not.
        # Don't increment for <s>, </s>, insert ops
        self.increment_source_loc = np.ones(num_ops, dtype=int)
        self.increment_source_loc[:2] = 0
        self.increment_source_loc[range(4, num_ops, 2)] = 0

        # If we have exhausted all source tokens, then we allow </s> and
        # insertion only.
        # In all cases, don't allow <s>.
        # Positions where mask is 1 will get -inf as the logit.
        self.end_mask = torch.ByteTensor([
            [1] + [0] * (num_ops - 1),
            # <s>, </s>, keep, delete
            np.concatenate(([1, 0, 1, 1],  self.increment_source_loc[4:]))
        ])
        if self._cuda:
            self.end_mask = self.end_mask.cuda()

        if self.use_code_state:
            self.state_h_proj = torch.nn.ModuleList(
                [nn.Linear(512, 256) for _ in range(2)])
            self.state_c_proj = torch.nn.ModuleList(
                [nn.Linear(512, 256) for _ in range(2)])

    def prepare_memory(self, io_embed, code_memory, _, ref_code):
        # code_memory:
        #   EncodedSequence, containing:
        #     mem: PackedSequencePlus,
        #          batch size * num pairs x code length x 512
        #     state: tuple containing two of
        #       2 (layers) * 2 (directions) x batch size * num pairs x 256
        #   or None
        pairs_per_example = io_embed.shape[1]
        ref_code = ref_code.cpu()
        if self.use_code_attn:
            # batch x code length x 512
            code_memory, code_lengths = code_memory.mem.pad(batch_first=True)
            code_mask = base.get_attn_mask(code_lengths, self._cuda)
            #code_memory = code_memory.unsqueeze(1).repeat(1, pairs_per_example,
            #                                              1, 1)
            #code_mask = code_mask.unsqueeze(1).repeat(1, pairs_per_example, 1)
            code_memory = utils.MaskedMemory(code_memory, code_mask)
        else:
            code_memory = code_memory.mem

        return LGRLRefineEditMemory(io_embed, code_memory, ref_code)

    def forward(self, io_embed, code_memory, _1, _2, dec_data):
        # io_embed: batch size x num pairs x 512
        # code_memory:
        #   EncodedSequence, containing:
        #     mem: PackedSequencePlus,
        #          batch size * num pairs x code length x 512
        #     state: tuple containing two of
        #       2 (layers) * 2 (directions) x batch size * num pairs x 256
        #   or None
        # dec_input:
        #   PackedSequencePlus, batch size x num ops x 4
        #      (op, emb pos, last token, io_embed index)
        # dec_output:
        #   PackedSequencePlus, batch size x num ops x 1 (op)
        batch_size, pairs_per_example = io_embed.shape[:2]

        if self.use_code_attn:
            state = self.init_state(code_memory, None, batch_size,
                    pairs_per_example)
            memory = self.prepare_memory(
                    io_embed, code_memory, None, dec_data.ref_code)
            #dec_input = dec_data.input.pad(batch_first=False)
            #for t in range(dec_input.shape[0]):
            #    dec_input_t = dec_input[t]
            #    self.decode_token(dec_input_t[2])

            memory.io = memory.io[list(dec_data.input.orig_to_sort)]
            memory.code = memory.code.apply(
                    lambda t: t[list(dec_data.input.orig_to_sort)])
            #lambda t: t[[
            #    exp_i
            #    for i in dec_data.input.orig_to_sort
            #    for exp_i in range(
            #        i * pairs_per_example,
            #        i * pairs_per_example + pairs_per_example)
            #]])

            logits = []
            offset = 0
            last_bs = 0
            batch_order = dec_data.input.orig_to_sort
            for i, bs in enumerate(dec_data.input.ps.batch_sizes):
                # Shape: bs x
                dec_data_slice = dec_data.input.ps.data[offset:offset + bs]
                if bs < last_bs:
                    memory.io = memory.io[:bs]
                    memory.code = memory.code.apply(lambda t: t[:bs])
                    batch_order = batch_order[:bs]
                    state = state.truncate(bs)

                state, logits_for_t = self.decode_token(
                        dec_data_slice[:, 0], state, memory, None, batch_order,
                        use_end_mask=False)
                logits.append(logits_for_t)
                offset += bs
                last_bs = bs

            logits = torch.cat(logits, dim=0)
            labels = dec_data.output.ps.data
            return logits, labels

        io_embed_flat = io_embed.view(-1, *io_embed.shape[2:])

        dec_input = dec_data.input.apply(lambda d: utils.maybe_concat(
            [
                self.op_embed(d[:, 0]),  # 256
                code_memory.mem.ps.data[d[:, 1]] if self.use_code_memory else None,  # 512
                self.last_token_embed(d[:, 2])],  # 256
            dim=1)).expand(pairs_per_example)

        dec_input = dec_input.apply(
                lambda d: torch.cat([d,
                    io_embed_flat[dec_data.io_embed_indices]], dim=1))

        state = self.init_state(code_memory, None, batch_size,
                pairs_per_example)
        state = (utils.flatten(state.h, 1), utils.flatten(state.c, 1))

        dec_output, _ = self.decoder(dec_input.ps, state)
        dec_output, _ = dec_output.data.view(-1, pairs_per_example,
                *dec_output.data.shape[1:]).max(dim=1)

        logits = self.out(dec_output)
        labels = dec_data.output.ps.data

        return logits, labels

    def decode_token(self, token, state, memory, attentions, batch_order=None,
            use_end_mask=True):
        pairs_per_example  = memory.io.shape[1]
        token_np = token.data.cpu().numpy()
        new_finished = state.finished.copy()

        # token shape: batch size (* beam size)
        # op_emb shape: batch size (* beam size) x 256
        op_emb = utils.expand(self.op_embed(token), pairs_per_example)

        # Relevant code_memory:
        # - same as last code_memory if we have insert, <s>, </s>
        # - incremented otherwise
        new_source_locs = state.source_locs + self.increment_source_loc[
                token_np]
        new_source_locs[state.finished] = 0

        if self.use_code_attn:
            code_memory = utils.flatten(state.context, 0)
        elif self.use_code_memory:
            code_memory_indices = memory.code.raw_index(
                    orig_batch_idx=range(len(state.source_locs)),
                    seq_idx=new_source_locs)
            code_memory = memory.code.ps.data[code_memory_indices]
            code_memory = utils.expand(code_memory, pairs_per_example)
        else:
            code_memory = None

        # Last token embedings
        # keep: token from appropriate source location
        #   (state.source_locs, not new_source_locs)
        # delete: UNK (ID 2)
        # insert, replace: the inserted/replaced token
        last_token_indices = np.zeros(len(token_np), dtype=int)
        keep_indices = []
        keep_orig_batch_indices = []
        keep_source_locs = []

        if batch_order is None:
            batch_order = range(len(token_np))
        for i, (batch_idx, t) in enumerate(zip(batch_order, token_np)):
            if t == 0:     # <s>
                last_token_indices[i] = t
            elif t == 1:     # </s>
                last_token_indices[i] = t
                new_finished[i] = True
            elif t == 2:  # keep
                keep_indices.append(i)
                keep_orig_batch_indices.append(batch_idx)
                keep_source_locs.append(state.source_locs[i])

                #last_token_indices.append(
                #        memory.ref_code.select(batch_idx,
                #            state.source_locs[batch_idx]).data[0])
                #last_token_indices.append(memory.source_tokens[batch_idx][
                #    state.source_locs[batch_idx]])
            elif t == 3: # delete
                last_token_indices[i] = 2 # UNK
            elif t >= 4:
                last_token_indices[i] = (t - 4) // 2
            else:
                raise ValueError(t)
        if keep_indices:
            last_token_indices[keep_indices] = memory.ref_code.select(
                keep_orig_batch_indices,
                [state.source_locs[i] for i in keep_indices]).data.numpy()

        last_token_indices = Variable(
                torch.LongTensor(last_token_indices))
        if self._cuda:
            last_token_indices = last_token_indices.cuda()
        last_token_emb = self.last_token_embed(last_token_indices)
        last_token_emb = utils.expand(last_token_emb, pairs_per_example)

        dec_input = utils.maybe_concat([op_emb, code_memory, last_token_emb,
            utils.flatten(memory.io, 0)], dim=1)

        dec_output, new_state = self.decoder(
                # 1 (time) x batch (* beam) * num pairs x hidden size
                dec_input.unsqueeze(0),
                # v before: 2 x batch (* beam) x num pairs x hidden
                # v after:  2 x batch (* beam) * num pairs x hidden
                (utils.flatten(state.h, 1), utils.flatten(state.c, 1)))
        new_state = (new_state[0].view_as(state.h),
                                 new_state[1].view_as(state.c))

        # shape after squeezing: batch size (* beam size) * num pairs x hidden
        dec_output = dec_output.squeeze(0)
        if self.use_code_attn:
            new_context, _ = self.code_attention(
                dec_output,
                utils.flatten(memory.code.memory, 0),
                utils.flatten(memory.code.attn_mask, 0))
            dec_output = utils.maybe_concat([dec_output, new_context], dim=1)
            new_context = new_context.view(
                    -1, pairs_per_example, new_context.shape[-1])
        else:
            new_context = None

        dec_output = dec_output.view(-1, pairs_per_example,
                                     *dec_output.shape[1:])
        dec_output, _ = dec_output.max(dim=1)

        # batch (* beam) x vocab size
        logits = self.out(dec_output)

        # If we have depleted the source tokens, then all we can do is end the
        # output or insert new tokens.
        # XXX use_end_mask must be false if state/tokens isn't in the same batch
        # order as memory.ref_code.
        if use_end_mask:
            end_mask = []
            for i, (loc, source_len) in enumerate(zip(new_source_locs,
                    memory.ref_code.orig_lengths())):
                if state.finished[i]:
                    end_mask.append(1)
                # source_len is 1 longer than in reality because we appended
                # </s> to each source sequence.
                elif loc == source_len - 1:
                    end_mask.append(1)
                elif loc >= source_len:
                    raise ValueError(loc)
                else:
                    end_mask.append(0)
            logits.data.masked_fill_(self.end_mask[end_mask], float('-inf'))

        return LGRLRefineEditDecoderState(new_source_locs, new_finished,
                new_context, *new_state), logits

    def postprocess_output(self, sequences, memory):
        ref_code = memory.ref_code.cpu()

        result = []

        ref_code_insert_locs = []
        ref_code_batch_indices = []
        ref_code_seq_indices = []

        for batch_idx, beam_outputs in enumerate(sequences):
            processed_beam_outputs = []
            for ops in beam_outputs:
                real_tokens = []
                source_index = 0
                for op in ops:
                    if op == 2:  # keep
                        ref_code_insert_locs.append(
                            (len(result), len(processed_beam_outputs),
                             len(real_tokens)))
                        ref_code_batch_indices.append(batch_idx)
                        ref_code_seq_indices.append(source_index)

                        real_tokens.append(-1)
                        source_index += 1
                    elif op == 3:  # delete
                        source_index += 1
                    elif op >= 4:
                        t = (op - 4) // 2
                        is_replace = (op - 4) % 2
                        real_tokens.append(t)
                        source_index += is_replace
                    else:
                        raise ValueError(op)
                processed_beam_outputs.append(real_tokens)
            result.append(processed_beam_outputs)

        if ref_code_insert_locs:
            tokens = ref_code.select(
                    ref_code_batch_indices,
                    ref_code_seq_indices).data.numpy()
            for (b, p,  s), t in zip(ref_code_insert_locs, tokens):
                result[b][p][s] = t

        return result

    def init_state(self, ref_code_memory, ref_trace_memory,
            batch_size, pairs_per_example):
        source_locs = np.zeros(batch_size, dtype=int)
        finished = np.zeros(batch_size, dtype=bool)

        if self.use_code_attn:
            context_size = (batch_size, pairs_per_example, 512)
            context = Variable(torch.zeros(*context_size))
            if self._cuda:
                context = context.cuda()
        else:
            context = None

        if self.use_code_state:
            new_state = []
            for s, proj in zip(ref_code_memory.state, (self.state_h_proj, self.state_c_proj)):
                # Shape: layers (2) x directions (2) x batch * pairs x hidden size
                s = s.contiguous().view(-1, 2, *s.shape[1:])
                # Shape: layers (2) x batch * pairs x directions (2) x hidden size
                s = s.permute(0, 2,  1, 3)
                # Shape: layers (2) x batch * pairs x directions * hidden size
                s = s.contiguous().view(*(s.shape[:2] + (-1,)))
                new_s = []
                for s_layer, proj_layer in zip(s, proj):
                    # Input: batch * pairs x directions * hidden size (=512)
                    # Output: batch * pairs x 256
                    new_s.append(proj_layer(s_layer))
                # Shape: 2 x batch * pairs x 256
                new_s = torch.stack(new_s)
                # Shape: 2 x batch x pairs x 256
                new_state.append(utils.unexpand(new_s, pairs_per_example, dim=1))
            return LGRLRefineEditDecoderState(
                    source_locs,
                    finished,
                    context,
                    *new_state)

        return LGRLRefineEditDecoderState(
                source_locs,
                finished,
                context,
                *utils.lstm_init(self._cuda, 2, 256, batch_size, pairs_per_example))


class LGRLKarel(nn.Module):
    def __init__(self, vocab_size, args):
        super(LGRLKarel, self).__init__()
        self.args = args

        self.encoder = karel_common.make_task_encoder(args)
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

        if self.args.karel_trace_enc.startswith('conv3d'):
            self.trace_encoder = TimeConvTraceEncoder(self.args)
        elif self.args.karel_trace_enc.startswith('lstm'):
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

        if self.args.karel_refine_dec == 'default':
            self.decoder = LGRLSeqRefineDecoder(vocab_size, args)
        elif self.args.karel_refine_dec == 'edit':
            self.decoder = LGRLSeqRefineEditDecoder(vocab_size, args)
        else:
            raise ValueError(self.args.karel_refine_dec)

        if self.args.karel_code_update_steps == 0:
            self.code_updater = None
        else:
            if self.args.karel_code_update == 'default':
                self.code_updater = CodeUpdater(args)
            else:
                raise ValueError(self.args.karel_code_update)

        if self.args.karel_io_enc == 'lgrl':
            self.encoder = karel_common.LGRLTaskEncoder(args)
        elif self.args.karel_io_enc == 'presnet':
            self.encoder = karel_common.PResNetTaskEncoder(args)
        else:
            raise ValueError(self.args.karel_io_enc)

    def encode(self, input_grid, output_grid, ref_code, ref_trace_grids,
               ref_trace_events, cag_interleave, code_update_info):
        pairs_per_example = input_grid.shape[1]

        # batch size x num pairs x 512
        io_embed = self.encoder(input_grid, output_grid)

        # PackedSequencePlus, batch size x length x 512
        ref_code_memory = self.code_encoder(ref_code)
        ref_code_memory = ref_code_memory.expand(pairs_per_example)

        # PackedSequencePlus, batch size x num pairs x length x  512
        ref_trace_memory = self.trace_encoder(ref_code_memory, ref_trace_grids,
                                              ref_trace_events, cag_interleave)

        # Perform some number of update steps.
        # TODO: Each update should use separate parameters?
        if self.args.karel_code_update_steps > 0:
            for i in range(self.args.karel_code_update_steps - 1, -1, -1):
                ref_code_memory = self.code_updater(
                        ref_code_memory, ref_trace_memory, code_update_info)
                if i > 0:
                    ref_trace_memory = self.trace_encoder(
                        ref_code_memory, ref_trace_grids, ref_trace_events,
                        cag_interleave)

        return io_embed, ref_code_memory, ref_trace_memory

    def decode(self, io_embed, ref_code_memory, ref_trace_memory, outputs,
            dec_data):
        return self.decoder(io_embed, ref_code_memory, ref_trace_memory,
                            outputs, dec_data)

    def decode_token(self, token, state, memory, attentions=None):
        return self.decoder.decode_token(token, state, memory, attentions)
