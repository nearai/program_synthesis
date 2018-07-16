from __future__ import absolute_import

import six

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from program_synthesis.common.modules import slicing_joining
from program_synthesis.common.modules import encoders


def remove_none(ls):
    return [item for item in ls if item is not None]


def concat_memories(memories, seq_lengths):
    assert len(memories) == len(seq_lengths)
    if len(memories) == len(seq_lengths) == 1:
        return memories[0], seq_lengths[0]

    summed_lengths = [sum(lengths) for lengths in zip(*seq_lengths)]
    max_length = max(summed_lengths)
    output = memories[0].data.new(memories[0].shape[0], max_length,
                                  memories[0].shape[2]).fill_(0)
    output = Variable(output, volatile=memories[0].volatile)
    for idx in range(memories[0].shape[0]):
        offset = 0
        for memory, seq_lengths_for_mem in zip(memories, seq_lengths):
            seq_length = seq_lengths_for_mem[idx]
            new_offset = offset + seq_length
            output[idx, offset:new_offset] = memory[idx, :seq_length]
            offset = new_offset

    return output, summed_lengths


class IOMixer(nn.Module):

    def __init__(self, args):
        super(IOMixer, self).__init__()
        self.num_units = args.num_units
        # TODO: Support bidirectional
        self.bidirectional = False  # args.bidirectional
        self._cuda = args.cuda
        self.io_count = args.io_count
        self.embed = nn.Embedding(256, self.num_units)
        self.input_encoder = nn.GRU(self.num_units, self.num_units, 1, batch_first=True,
                                    bidirectional=self.bidirectional)
        self.output_encoder = nn.GRU(self.num_units, self.num_units, 1, batch_first=True,
                                     bidirectional=self.bidirectional)
        self.mixer = nn.Linear(2*self.io_count*self.num_units, self.num_units)

    def forward(self, input_keys_embed, inputs, arg_nums, outputs):
        inputs = self.embed(inputs)
        outputs = self.embed(outputs)
        input_keys = input_keys_embed.unsqueeze(0)
        _, inp_hidden = self.input_encoder(inputs, input_keys)
        num_directions = 2 if self.bidirectional else 1
        init = Variable(torch.zeros(
            num_directions, outputs.size(0), self.num_units))
        if self._cuda:
            init = init.cuda()
        _, out_hidden = self.output_encoder(outputs, init)
        if self.bidirectional:
            # TODO: Support bidirectional...
            assert False
        else:
            inp_hidden = inp_hidden.squeeze(0)
            out_hidden = out_hidden.squeeze(0)

        all_inputs = list(slicing_joining.split(inp_hidden, arg_nums, dim=0))
        aggregated_inputs = [torch.mean(t, dim=0) for t in all_inputs]
        batch_size = out_hidden.size(0) / self.io_count
        aggregated_inputs = torch.cat(aggregated_inputs, dim=0).view(batch_size, -1)

        out_hidden = out_hidden.view(batch_size, -1)
        inp_out = torch.cat([aggregated_inputs, out_hidden], dim=1)
        return F.relu(self.mixer(inp_out))


class SpecEncoder(nn.Module):

    def __init__(self, vocab_size, args, embed=None):
        super(SpecEncoder, self).__init__()
        self.num_units = args.num_units
        proj_inputs = 0
        if args.read_text:
            self.text_encoder = encoders.SequenceEncoder(vocab_size, args, embed=embed)
            proj_inputs += 1
        if args.read_io:
            self.io_encoder = IOMixer(args)
            proj_inputs += 1

        self.proj = nn.Linear(proj_inputs * self.num_units, self.num_units)

    def forward(self, text_enc, text_memory, text_lengths, io_enc):
        # text_enc: batch size x num units
        # text_memory: torch.nn.utils.rnn.PackedSequence
        hidden = torch.cat(remove_none([text_enc, io_enc]), dim=1)
        memories = remove_none([text_memory])
        seq_lengths = remove_none([text_lengths])
        if memories:
            memory, seq_lengths = concat_memories(memories, seq_lengths)
        else:
            memory = io_enc.unsqueeze(1)
            seq_lengths = [1] * io_enc.shape[0]

        return self.proj(hidden), memory, seq_lengths

    @property
    def output_dim(self):
        return self.num_units
