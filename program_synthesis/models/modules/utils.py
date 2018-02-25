import collections

import torch
from torch.autograd import Variable


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


class SequenceMemory(
        collections.namedtuple('SequenceMemory', ['mem', 'state'])):
    def expand(self, k):
        mem = None if self.mem is None else self.mem.expand(k)
        # Assumes state is for an LSTM.
        state = None if self.state is None else tuple(
                expand(state_elem, k, dim=1) for state_elem in self.state)
        return SequenceMemory(mem, state)
