import collections
import functools

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from program_synthesis.common.modules import attention
from program_synthesis.common.models import beam_search

from program_synthesis.karel.dataset import data
from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.models.modules import karel
from program_synthesis.karel.models.modules import karel_common
from program_synthesis.karel.models.modules import utils


class ScaledDotProductPointer(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(ScaledDotProductPointer, self).__init__()
        self.query_proj = nn.Linear(query_dim, key_dim)
        self.temper = np.power(key_dim, 0.5)

    def forward(self, query, keys, attn_mask=None):
        # query shape: batch x query dim
        # keys shape: batch x num keys x key dim
        # attn_mask shape: batch x num keys

        if attn_mask is not None:
            # batch x 1 x num keys
            attn_mask = attn_mask.unsqueeze(1)

        # batch x 1 x query dim
        query = query.unsqueeze(1)
        # batch x 1 x key dim
        query = self.query_proj(query)
        # batch x key dim x num keys
        keys = keys.transpose(1, 2)

        # batch x 1 x num keys
        attn = torch.bmm(query, keys) / self.temper
        attention.maybe_mask(attn, attn_mask)
        # batch x num keys
        attn = attn.squeeze(1)

        return attn


class KarelEdit(nn.Module):
    initial_vocab = data.Vocab(mutation.ACTION_NAMES + ('if', 'while',
                                                        'repeat', 'ifElse'))
    choice_vocab = data.Vocab(initial_vocab.keys + tuple(
        range(len(mutation.CONDS) + len(mutation.REPEAT_COUNTS))))

    class Memory(
            collections.namedtuple('Memory', ('io', 'current_grid',
                                              'current_code')),
            beam_search.BeamSearchMemory):
        # io: batch x num pairs x hidden size
        # current_grid: batch x num pairs x hidden size
        # current_code: MaskedMemory, batch x code length x hidden size

        def expand_by_beam(self, beam_size):
            io = utils.expand(self.io, beam_size)
            current_grid = utils.expand(self.current_grid, beam_size)
            current_code = self.current_code.expand_by_beam(beam_size)
            return Memory(io, current_grid, current_code)

        def to_flat(self):
            return (self.io, self.current_grid, self.current_code.memory,
                    self.current_code.attn_mask)

        @classmethod
        def from_flat(cls, io, current_grid, current_code_memory,
                      current_code_attn_mask):
            return cls(io, current_grid,
                       utils.MaskedMemory(current_code_memory,
                                          current_code_attn_mask))

    State = utils.MultiContextLSTMState

    def __init__(self, vocab_size, args):
        super(KarelEdit, self).__init__()
        self._cuda = args.cuda

        self.use_code_attn = True
        self.use_code_state = True

        self.io_encoder = karel_common.make_task_encoder(args)
        self.grid_encoder = karel_common.make_grid_encoder(args)

        if args.karel_code_enc == 'default':
            self.code_encoder = karel.CodeEncoder(vocab_size, args)
        else:
            raise ValueError(args.karel_code_enc)

        logits_in = 256 + (512 if self.use_code_attn else 0)
        if args.karel_merge_io == 'max':
            self.merge_io_pairs = lambda emb: torch.max(emb, dim=1)[0]
        elif args.karel_merge_io == 'setlstm':
            self.process_set = utils.ProcessSet(
                input_size=logits_in,
                hidden_size=logits_in,
                num_layers=1,
                num_steps=5)
            # emb shape: batch size x num pairs x input size
            self.merge_io_pairs = lambda emb: self.process_set(
                    utils.MaskedMemory(emb, None))

        # ADD_ACTION
        # move, turn{Left,Right}, {pick,put}Marker
        num_actions = 5
        # WRAP_BLOCK
        # WRAP_IFELSE
        num_block_types = 4
        num_conds = len(mutation.CONDS)
        num_repeats = len(mutation.REPEAT_COUNTS)

        self.bos_emb = nn.Parameter(torch.Tensor(1, 256))
        self.bos_emb.data.normal_(0, 1)

        self.choice_emb = nn.Embedding(
            num_embeddings=num_actions +  # step 1, adding an action
            num_block_types +  # step 1, wrapping with a block
            num_conds +  # step 2, if/ifElse/while
            num_repeats,  # step 2, repeat
            embedding_dim=256)

        # + 1 for "no choices"
        self.initial_logits = nn.Linear(logits_in,
                                        num_actions + num_block_types + 1)
        self.cond_logits = nn.Linear(logits_in, num_conds)
        self.repeat_logits = nn.Linear(logits_in, num_repeats)
        self.pointer_logits = ScaledDotProductPointer(
            query_dim=logits_in, key_dim=512)

        # LSTM
        lstm_input = 256
        if self.io_encoder is not karel_common.none_fn:
            lstm_input += 512
        if self.grid_encoder is not karel_common.none_fn:
            lstm_input += 256
        if self.use_code_attn:
            lstm_input += 512
        self.decoder = nn.LSTM(
            input_size=lstm_input, hidden_size=256, num_layers=2)
        if self.use_code_attn:
            self.code_attention = attention.SimpleSDPAttention(
                query_dim=256, values_dim=256 * 2)
        if self.use_code_state:
            self.proj_decoder = utils.ProjectLSTMState(
                in_hidden_size=256,
                out_hidden_size=256,
                num_layers=2,
                num_directions=2)
        self.proj_code_for_input = nn.Linear(512, 256)

    def prepare_initial(self, input_grids, output_grids, current_grids,
                        current_code):
        batch_size = input_grids.shape[0]
        pairs_per_example = input_grids.shape[1]

        io_embed = self.io_encoder(input_grids, output_grids)
        current_grid_embed = self.grid_encoder(current_grids)
        current_code_enc = self.code_encoder(current_code)

        current_code_memory = utils.MaskedMemory.from_psp(current_code_enc.mem)
        current_code_memory = utils.MaskedMemory(
            current_code_memory.memory.unsqueeze(1).expand(
                -1, pairs_per_example, -1, -1),
            current_code_memory.attn_mask.unsqueeze(1).expand(
                -1, pairs_per_example, -1))

        memory = KarelEdit.Memory(io_embed, current_grid_embed,
                                  current_code_memory)
        state = self.init_state(batch_size, pairs_per_example,
                                current_code_enc)

        emb_for_logits, state = self.step(
            self.bos_emb.expand(batch_size, -1), state, memory)
        # batch x (num actions + num block types)
        logits = self.initial_logits(emb_for_logits)

        return state, memory, logits

    def step(self, input_emb, state, memory):
        # input_emb shape: batch x hidden size
        # TODO: Allow input_emb to be batch x num pairs x hidden size

        batch_size = state.h.shape[1]
        pairs_per_example = state.h.shape[2]

        # shape: batch x num pairs x hidden size
        dec_input = utils.maybe_concat(
            [
                memory.io, memory.current_grid, state.context,
                input_emb.unsqueeze(1).expand(-1, pairs_per_example, -1)
            ],
            dim=2)
        # batch * num pairs x hidden size
        dec_input = dec_input.view(-1, dec_input.shape[-1])
        dec_output, new_state = self.decoder(
            # 1 x batch * num pairs x hidden size
            dec_input.unsqueeze(0),
            # v before: 2 x batch x num pairs x hidden
            # v after:  2 x batch * num pairs x hidden
            (utils.flatten(state.h, 1), utils.flatten(state.c, 1)))
        # batch * num pairs x hidden size
        dec_output = dec_output.squeeze(0)

        # Create new state
        if self.use_code_attn:
            new_context, _ = self.code_attention(
                dec_output,
                utils.flatten(memory.current_code.memory, 0),
                utils.flatten(memory.current_code.attn_mask, 0))
            new_context = new_context.view_as(state.context)
        else:
            new_context = None
        state = KarelEdit.State(new_context, new_state[0].view_as(state.h),
                                new_state[1].view_as(state.c))

        # Process output
        # batch x num pairs x hidden size
        dec_output = dec_output.view(batch_size, pairs_per_example, -1)
        # batch x num pairs x hidden size
        emb_for_logits = utils.maybe_concat((new_context, dec_output), dim=2)
        # batch x hidden
        emb_for_logits = self.merge_io_pairs(emb_for_logits)

        return emb_for_logits, state

    def init_state(self, batch_size, pairs_per_example, current_code_enc):
        if self.use_code_attn:
            context_size = (batch_size, pairs_per_example, 512)
            context = Variable(torch.zeros(*context_size))
            if self._cuda:
                context = context.cuda()
        else:
            context = None

        # TODO: Deduplicate with LGRLRefineEditDecoder and LGRLRefineDecoder.
        if self.use_code_state:
            # Assumes that we have batch * num pairs states
            #new_state = [
            #    utils.unexpand(
            #        s, pairs_per_example, dim=1)
            #    for s in self.proj_decoder(current_code_enc.state)
            #]
            new_state = [
                # s before: layers * directions x batch x hidden
                # s after:  layers * directions x batch x num pairs x hidden
                s.unsqueeze(2).expand(-1, -1, pairs_per_example, -1)
                for s in self.proj_decoder(current_code_enc.state)
            ]
            return KarelEdit.State(context, *new_state)

        return KarelEdit.State(context, *utils.lstm_init(
            self._cuda, 2, 256, batch_size, pairs_per_example))

    # Methods for TorchFold.
    def __getattr__(self, name):
        if name.startswith('tf_get_log_prob'):
            return self.tf_get_log_prob
        elif name.startswith('tf_torch_log_softmax'):
            return functools.partial(torch.nn.functional.log_softmax, dim=-1)
        else:
            return super(KarelEdit, self).__getattr__(name)

    def tf_torch_zero(self):
        v = Variable(torch.Tensor([0]))
        if self._cuda:
            v = v.cuda()
        return v

    def tf_step(self, input_emb, state_context, state_h, state_c, *memory):
        assert len(memory) == 4
        # state_{h,c} before: batch x num layers x num pairs x hidden
        # state_{h,c} after:  num layers x batch x num pairs x hidden
        state_h = state_h.permute(1, 0, 2, 3)
        state_c = state_c.permute(1, 0, 2, 3)
        emb_for_logits, state = self.step(
            input_emb,
            KarelEdit.State(state_context, state_h, state_c),
            KarelEdit.Memory.from_flat(*memory))

        new_state_h = state.h.permute(1, 0, 2, 3)
        new_state_c = state.c.permute(1, 0, 2, 3)
        return emb_for_logits, state.context, new_state_h, new_state_c

    def tf_get_code_emb(self, memory, loc):
        # memory: batch x length x hidden
        # loc: batch, LongTensor
        assert (loc < memory.shape[1]).all()
        return self.proj_code_for_input(memory[range(memory.shape[0]), loc])

    def tf_get_log_prob(self, log_probs, idx):
        # log_probs: batch x max length
        # idx: batch, LongTensor
        assert (idx < log_probs.shape[1]).all()
        log_probs_cpu = log_probs.cpu()
        idx_cpu = idx.cpu()
        return log_probs[range(log_probs.shape[0]), idx]

    def tf_batched_sum(self, v1, v2, v3, v4, v5):
        return torch.sum(torch.stack((v1, v2, v3, v4, v5), dim=-1), dim=-1)


# Example:
#  Batch item 1
#   move <pos 2>
#   pickMarker <pos 2>
#   pickMarker <pos 3>
#   if <pos 2> <pos 4>
#   ifElse <pos 3> <pos 4> <pos 4>
#  Batch item 2
#   putMarker <pos 2>
#   putMarker <pos 3>
#   if <pos 2> <pos 4>
#
# After step 1, we have:
#  state: (abstractly) 2 x num pairs x hidden size
#   Out of this, we want:
#    4x times batch item 0 (move, pickMarker, if, ifElse)
#    2x times batch item 1 (putMarker, if)
#   We can get this by applying:
#    state.repeat([4, 2], dim=0)
#  logits: 2 x 9
#   Out of this, we want:
#    Batch item 1
#     (0, move)
#     (0, pickMarker)
#     (0, if),
#     (0, ifElse)
#    Batch item 2
#     (1, putMarker)
#     (1, if)
#   We can get this by applying advanced indexing:
#     logits[np.arange(2).repeat([4, 2]),
#            [move, pickMarker, if, ifElse, putMarker, if]]
#
# Step 2: separate into three parts
