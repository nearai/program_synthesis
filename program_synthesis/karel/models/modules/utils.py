import collections

import torch
from torch import nn
from torch.autograd import Variable

from program_synthesis.common.modules import attention
from program_synthesis.common.models import beam_search


def default(value, if_none):
    return if_none if value is None else value


def expand(v, k, dim=0):
    # Input: ... x d_dim x ...
    # Output: ... x d_dim * k x ... where
    #   out[..., 0] = out[..., 1] = ... out[..., k],
    #   out[..., k + 0] = out[..., k + 1] = ... out[..., k + k],
    # and so on.
    dim += 1
    return v.unsqueeze(dim).repeat(
        *([1] * dim + [k] + [1] *
          (v.dim() - dim))).view(*(v.shape[:dim - 1] + (-1, ) + v.shape[dim:]))


def unexpand(v, k, dim=0):
    # Input: ... x d_dim x ...
    # Output: ... x  d_dim / k x k x ...
    return v.view(*(v.shape[:dim] + (-1, k) + v.shape[dim+1:]))


def flatten(v, k):
    # Input: d1 x ... x dk x dk+1 x ... x dn
    # Output: d1 x ... x dk * dk+1 x ... x dn
    args = v.shape[:k] + (-1, ) + v.shape[k + 2:]
    return v.contiguous().view(*args)


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


def repeat(tensor, repeats, dim):
    '''Equivalent of numpy.repeat for Torch tensors.

    `repeats` must be a 1D Torch LongTensor with as many elements as the size of
    `tensor` in `dim`.'''
    index = torch.from_numpy(np.arange(tensor.shape[dim]).repeat(repeats))
    if isinstance(tensor, Variable):
        index = Variable(index)
    if tensor.is_cuda:
        index = index.cuda()

    return torch.index_select(tensor, dim, index)


def logsumexp(tensor):
    m = torch.max(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m)))


def set_pop(s, value):
    if value in s:
        s.remove(value)
        return True
    return False


def lstm_init(cuda, num_layers, hidden_size, *batch_sizes):
    init_size = (num_layers, ) + batch_sizes + (hidden_size, )
    init = Variable(torch.zeros(*init_size))
    if cuda:
        init = init.cuda()
    return (init, init)


class EncodedSequence(
        collections.namedtuple('EncodedSequence', ['mem', 'state'])):

    def expand(self, k):
        mem = None if self.mem is None else self.mem.expand(k)
        # Assumes state is for an LSTM.
        state = None if self.state is None else tuple(
                expand(state_elem, k, dim=1) for state_elem in self.state)
        return EncodedSequence(mem, state)


class MaskedMemory(
        collections.namedtuple('MaskedMemory', ['memory', 'attn_mask'])):
    @classmethod
    def from_psp(cls, psp, new_shape=None):
        memory, lengths = psp.pad(batch_first=True)
        mask = get_attn_mask(lengths, memory.is_cuda)

        if new_shape is not None:
            memory = memory.view(*(new_shape + memory.shape[1:]))
            mask = mask.view(*(new_shape + mask.shape[1:]))

        return cls(memory, mask)

    def expand_by_beam(self, beam_size):
        return MaskedMemory(*(v.unsqueeze(1).repeat(1, beam_size, *([1] * (
            v.dim() - 1))).view(-1, *v.shape[1:]) for v in self))

    def apply(self, fn):
        return MaskedMemory(fn(self.memory), fn(self.attn_mask))


def get_attn_mask(seq_lengths, cuda):
    max_length, batch_size = max(seq_lengths), len(seq_lengths)
    ranges = torch.arange(
        0, max_length,
        out=torch.LongTensor()).unsqueeze(0).expand(batch_size, -1)
    attn_mask = (ranges >= torch.LongTensor(seq_lengths).unsqueeze(1))
    if cuda:
        attn_mask = attn_mask.cuda()
    return attn_mask


class MultiContextLSTMState(
        collections.namedtuple('MultiContextLSTMState', ['context', 'h', 'c']),
        beam_search.BeamSearchState):
    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        selected = [
            None if self.context is None else self.context.view(
                batch_size, -1,
                *self.context.shape[1:])[indices.data.numpy()]
        ]
        for v in self.h, self.c:
            # before: 2 x batch size (* beam size) x num pairs x hidden
            # after:  2 x batch size x beam size x num pairs x hidden
            v = v.view(2, batch_size, -1, *v.shape[2:])
            # result: 2 x indices.shape[1] x num pairs x hidden
            selected.append(v[(slice(None), ) + tuple(indices.data.numpy(
            ))])
        return MultiContextLSTMState(*selected)

    def truncate(self, k):
        return MultiContextLSTMState(self.context[:k], self.h[:, :k], self.c[:, :k])


# Limitations:
# - Only supports LSTM states
# - Only supports unidirectional output state
class ProjectLSTMState(nn.Module):
    def __init__(self, in_hidden_size, out_hidden_size, num_layers,
                 num_directions):
        super(ProjectLSTMState, self).__init__()

        self.num_layers = num_layers
        self.num_directions = num_directions

        self.state_h_proj = nn.ModuleList([
            nn.Linear(in_hidden_size * num_directions, out_hidden_size)
            for _ in range(num_layers)
        ])
        self.state_c_proj = nn.ModuleList([
            nn.Linear(in_hidden_size * num_directions, out_hidden_size)
            for _ in range(num_layers)
        ])

    def forward(self, state):
        new_state =  []

        for s, proj in zip(state, (self.state_h_proj, self.state_c_proj)):
            # Shape: layers x directions x batch x in hidden size
            s = s.contiguous().view(-1, self.num_directions, *s.shape[1:])
            # Shape: layers x batch x directions x in hidden size
            s = s.permute(0, 2,  1, 3)
            # Shape: layers x batch x directions * in hidden size
            s = s.contiguous().view(*(s.shape[:2] + (-1,)))
            new_s = []
            for s_layer, proj_layer in zip(s, proj):
                # Input: batch x directions * in hidden size
                # Output: batch x out hidden size
                new_s.append(proj_layer(s_layer))
            # Shape: num layers x batch x out hidden size
            new_s = torch.stack(new_s)
            new_state.append(new_s)

        return new_state


class ProcessSet(nn.Module):
    '''Similar to "process" block in https://arxiv.org/abs/1511.06391v4,
    section 4.3.'''

    def __init__(self, input_size, hidden_size, num_layers, num_steps):
        super(ProcessSet, self).__init__()
        self.num_steps = num_steps

        self.process = nn.LSTM(input_size, hidden_size, num_layers)
        self.attn = attention.SimpleSDPAttention(
            query_dim=hidden_size, values_dim=input_size)

        self.initial_input = nn.Parameter(torch.Tensor(1, input_size))
        self.initial_input.data.normal_(0, 1)

    def forward(self, inputs):
        # inputs: MaskedMemory, shape batch size x input size
        state = None
        process_input = self.initial_input.expand(inputs.memory.shape[0], -1)

        for i in range(self.num_steps):
            # output shape:
            output, state = self.process(process_input.unsqueeze(0), state)
            output = output.squeeze(0)

            process_input, _ = self.attn(
                query=output, values=inputs.memory, attn_mask=inputs.attn_mask)

        return process_input
