import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class SeqEncoder(nn.Module):

    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.num_units = args.num_units
        self.num_encoder_layers = args.num_encoder_layers
        self.bidirectional = args.bidirectional
        self._cuda = args.cuda
        self.encoder = nn.GRU(
            self.num_units, self.num_units, args.num_encoder_layers, batch_first=True,
            dropout=args.encoder_dropout, bidirectional=self.bidirectional)
        directions = 2 if self.bidirectional else 1
        if directions * args.num_encoder_layers > 1:
            self.encoder_proj = nn.Linear(directions * args.num_encoder_layers * self.num_units, self.num_units)

    def forward(self, masked_padded_texts, text_lengths):
        # Prepare initial state for the encoder.
        batch_size = masked_padded_texts.shape[0]
        num_directions = 2 if self.bidirectional else 1
        t_type = (torch.cuda.FloatTensor if self._cuda else torch.FloatTensor)
        init = Variable(t_type(num_directions * self.num_encoder_layers, batch_size, self.num_units).fill_(0),
                        volatile=not self.training)

        # Prepare the input.
        masked_packed_texts = pack_padded_sequence(masked_padded_texts, text_lengths, batch_first=True)

        # memory: torch.nn.utils.rnn.PackedSequence
        # [bsz x len x (dim * num_directions)]
        memory, hidden = self.encoder(masked_packed_texts, init)
        memory, _ = pad_packed_sequence(memory, batch_first=True)
        if num_directions * self.num_encoder_layers > 1:
            # Make batch-first
            hidden = hidden.transpose(0, 1).contiguous()
            # Project to num_units units
            hidden = self.encoder_proj(hidden.view(hidden.shape[0], -1))
        else:
            hidden = hidden.squeeze(0)
        return hidden, memory
