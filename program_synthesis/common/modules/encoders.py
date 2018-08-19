import torch
import torch.nn as nn

from program_synthesis.common.tools.packed_sequence import PackedSequencePlus


class SequenceEncoder(nn.Module):

    def __init__(self, vocab_size, args, embed=None):
        super(SequenceEncoder, self).__init__()
        self.args = args
        if embed is None:
            self.embed = nn.Embedding(vocab_size, args.num_units)
        else:
            self.embed = embed
        self.directions = (
            2 if args.encoder_bidirectional else 1) * args.num_encoder_layers
        self.encoder = nn.GRU(
            args.num_units, args.num_units, args.num_encoder_layers, batch_first=True,
            dropout=args.encoder_dropout if args.num_encoder_layers > 1 else 0,
            bidirectional=args.encoder_bidirectional
        )

    def forward(self, inputs):
        if isinstance(inputs, PackedSequencePlus):
            encoded = inputs.apply(self.embed).ps
        else:
            encoded = self.embed(inputs)
        memory, hidden = self.encoder(encoded)
        if self.directions > 1:
            hidden = hidden.transpose(0, 1).view(
                inputs.size(0), -1).contiguous()
        else:
            hidden = hidden.squeeze(0)
        if isinstance(inputs, PackedSequencePlus):
            memory = inputs.with_new_ps(memory)
        return hidden, memory

    @property
    def output_dim(self):
        return self.directions * self.args.num_units
