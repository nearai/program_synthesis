import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from program_synthesis.common.modules import encoders
from program_synthesis.common.modules import decoders


class Sequence2Sequence(nn.Module):

    def __init__(self, input_vocab_size, output_vocab_size, args, encoder_cls=encoders.SequenceEncoder):
        super(Sequence2Sequence, self).__init__()
        self.args = args
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.input_embed = nn.Embedding(input_vocab_size, args.num_units)
        if args.share_embeddings:
            assert input_vocab_size == output_vocab_size
            self.output_embed = self.input_embed
        else:
            self.output_embed = nn.Embedding(output_vocab_size, args.num_units)
        self.encoder = encoder_cls(input_vocab_size, args, embed=self.input_embed)
        self.decoder = decoders.get_decoder_cls(args.seq2seq_decoder)(
            output_vocab_size, self.encoder.output_dim, args, embed=self.output_embed)

    def forward(self, inputs, outputs):
        encoding, memory = self.encoder(inputs)
        # TODO: Handle memory that is packed sequence
        return self.decoder(encoding, memory, outputs)

    def decode(self, encoding, memory, outputs):
        return self.decoder(encoding, memory, outputs)

    def sample(self, inputs):
        encoding, memory = self.encoder(inputs)
        return self.decoder.sample(encoding, memory)

    def decode_token(self, prev_step, hidden, memory_attn_mask, attentions=None):
        return self.decoder.decode_token(prev_step, hidden, memory_attn_mask, attentions)
