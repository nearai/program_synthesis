"""Functions for slicing, joining, padding, packing. Most of the code is copied from the upcoming PyTorch release and
should be removed once the new version becomes available.
"""
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


# PyTorch code. Should be removed once the new version becomes available

def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sequences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = Variable(torch.ones(25, 300))
        >>> b = Variable(torch.ones(22, 300))
        >>> c = Variable(torch.ones(15, 300))
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Variable of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable


def pack_sequence(sequences):
    r"""Packs a list of variable length Variables

    ``sequences`` should be a list of Variables of size ``Lx*``, where L is
    the length of a sequence and * is any number of trailing dimensions,
    including zero. They should be sorted in the order of decreasing length.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = Variable(torch.Tensor([1,2,3]))
        >>> b = Variable(torch.Tensor([4,5]))
        >>> c = Variable(torch.Tensor([6]))
        >>> pack_sequence([a, b, c]])
        PackedSequence(data=
         1
         4
         6
         2
         5
         3
        [torch.FloatTensor of size 6]
        , batch_sizes=[3, 2, 1])


    Arguments:
        sequences (list[Variable]): A list of sequences of decreasing length.

    Returns:
        a :class:`PackedSequence` object
    """
    return pack_padded_sequence(pad_sequence(sequences), [v.size(0) for v in sequences])


def split(tensor, split_size_or_sections, dim=0):
    """Splits the tensor into chunks.
    If ``split_size_or_sections`` is an integer type, then ``tensor`` will be
    split into equally sized chunks (if possible).
    Last chunk will be smaller if the tensor size along a given dimension
    is not divisible by ``split_size``.
    If ``split_size_or_sections`` is a list, then ``tensor`` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in ``dim`` according
    to ``split_size_or_sections``.

    Arguments:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
        list of sizes for each chunk
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)

    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        num_splits = (dim_size + split_size - 1) // split_size
        last_split_size = split_size - (split_size * num_splits - dim_size)

        def get_split_size(i):
            return split_size if i < num_splits - 1 else last_split_size
        return tuple(tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i))) for i
                     in range(0, num_splits))

    else:
        if dim_size != sum(split_size_or_sections):
            raise ValueError("Sum of split sizes exceeds tensor dim")
        split_indices = [0] + split_size_or_sections
        split_indices = torch.cumsum(torch.Tensor(split_indices), dim=0)

        return tuple(
            tensor.narrow(int(dim), int(start), int(length))
            for start, length in zip(split_indices, split_size_or_sections))

# The following methods allow running RNN on sequences that are not ordered by their length.


def _array_to_tensor(arr, cuda):
    t = torch.LongTensor(arr)
    if cuda:
        t = t.cuda()
    return Variable(t)


def _forward_and_reverse_indices(sequence):
    forward_indices = list(np.argsort(sequence))[::-1]
    reverse_indices = sorted(enumerate(forward_indices), key=lambda x: x[1])
    reverse_indices = list(zip(*reverse_indices)[0])
    return forward_indices, reverse_indices


def run_padded_rnn(sequences, rnn, init_state, seq_lengths, cuda):
    """ Runs rnn on the padded but unordered sequences.

    :param sequences: A tensor of shape batch_size x max_seq_length x dim;
    :param rnn: instance of nn.GRU;
    :param init_state: tensor num_layers x batch_size x hidden_size;
    :param seq_lengths: a list with batch_size integers <= max_seq_length;
    :param cuda: bool;
    :return: Variable num_layers x batch_size x hidden_size;
    """

    # We use pack_sequence which requires sequences sorted by their length in desc order.
    forward_indices, reverse_indices = _forward_and_reverse_indices(seq_lengths)

    ordered_sequences = torch.index_select(sequences, dim=0, index=_array_to_tensor(forward_indices, cuda))
    packed_sequences = pack_padded_sequence(ordered_sequences, sorted(seq_lengths, key=lambda x: -x), batch_first=True)
    _, state = rnn(packed_sequences, init_state)
    return torch.index_select(state, dim=1, index=_array_to_tensor(reverse_indices, cuda))


def run_rnn(sequences, rnn, init_state, cuda):
    """ Runs rnn on a list of unordered sequences.

    :param sequences: A list of batch_size elements each with shape seq_lengths[i] x dim;
    :param rnn: instance of nn.GRU;
    :param init_state: tensor num_layers x batch_size x hidden_size;
    :param cuda: bool;
    :return: Variable num_layers x batch_size x hidden_size;
    """

    seq_lengths = [v.size(0) for v in sequences]
    # We use pack_sequence which requires sequences sorted by their length in desc order.
    forward_indices, reverse_indices = _forward_and_reverse_indices(seq_lengths)

    ordered_sequences = [sequences[i] for i in forward_indices]
    packed_sequences = pack_sequence(ordered_sequences)
    _, state = rnn(packed_sequences, init_state)
    return torch.index_select(state, dim=1, index=_array_to_tensor(reverse_indices, cuda))
