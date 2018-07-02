from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.models import prepare_spec

from program_synthesis.common.modules import decoders
from program_synthesis.algolisp.models.modules import encoders


class Seq2Seq(nn.Module):

    def __init__(self, inp_vocab_size, out_vocab_size, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.inp_embed = nn.Embedding(inp_vocab_size + args.num_placeholders, args.num_units)
        self.encoder = nn.GRU(args.num_units, args.num_units, args.num_encoder_layers, batch_first=True)
        # Same vocab flag
        self.decoder = decoders.SeqDecoder(out_vocab_size, args)

    def encode(self, inputs):
        emb_inputs = self.inp_embed(data.replace_pad_with_end(inputs))
        init = Variable(torch.zeros(args.num_encoder_layers, emb_inputs.size(0), self.args.num_units))
        if self.args.cuda:
            init = init.cuda()
        _, hidden = self.encoder(emb_inputs, init)
        return hidden

    def forward(self, inputs, outputs):
        hidden = self.encode(inputs)
        return self.decoder(data.replace_pad_with_end(outputs), init=hidden)

    def sample(self, inputs, sampler=decoders.argmax_sampler):
        hidden = self.encode(inputs)
        return self.decoder.sample(hidden=hidden, sampler=sampler)


class Seq2SeqAttn(nn.Module):

    def __init__(self, word_vocab_size, code_vocab_size, args):
        super(Seq2SeqAttn, self).__init__()
        self.args = args
        self.num_units = args.num_units
        self.num_placeholders = args.num_placeholders
        self.bidirectional = args.bidirectional
        self._cuda = args.cuda
        self.word_embed = nn.Embedding(
            word_vocab_size + self.num_placeholders, self.num_units)
        self.code_embed = nn.Embedding(
            code_vocab_size + self.num_placeholders, self.num_units)
        self.encoder = encoders.SpecEncoder(args)
        num_directions = 2 if self.bidirectional else 1
        mem_dim = self.num_units * num_directions
        DECODERS = {
            'attn_decoder': decoders.SeqDecoderAttn,
            'multi_attn_decoder': decoders.SeqDecoderMultiAttn,
            'past_attn_decoder': decoders.SeqDecoderPastAttn,
            'luong_attn_decoder': decoders.SeqDecoderAttnLuong,
        }
        self.decoder = DECODERS[args.seq2seq_decoder](
            code_vocab_size, mem_dim, args, embed=self.code_embed)

    def encode_text(self, inputs):
        # inputs: PackedSequencePlus
        return self.encoder.text_encoder(inputs.apply(self.word_embed))

    def encode_io(self, input_keys, inputs, arg_nums, outputs):
        input_keys_embed = self.code_embed(input_keys)
        return self.encoder.io_encoder(input_keys_embed, inputs, arg_nums, outputs)

    def encode_code(self, code_seqs):
        # code_seqs: PackedSequencePlus
        return self.encoder.code_encoder(code_seqs.apply(self.code_embed))

    def encode_trace(self, prepared_trace):
        return self.encoder.trace_encoder(prepared_trace)

    def extend_tensors(
            self, code_info, batch_size, batch_ids):
        # TODO: should be a separate module probably with its parameters.
        if code_info:
            code_enc, code_memory, orig_seq_lengths  = code_info

            # Every item in the batch has code.
            if len(batch_ids) == batch_size:
                return code_enc, code_memory, orig_seq_lengths

            # Otherwise, stagger empty encodings/memories with real ones
            enc_to_stack = [self.empty_candidate_code_hidden] * batch_size
            memory_to_stack = [torch.zeros_like(code_memory[0])] * batch_size
            seq_lengths = [0] * batch_size

            for i, batch_id in enumerate(batch_ids):
                enc_to_stack[batch_id] = code_enc[i]
                memory_to_stack[batch_id] = code_memory[i]
                seq_lengths[batch_id] = orig_seq_lengths[i]

            enc = torch.stack(enc_to_stack)
            memory = torch.stack(memory_to_stack)
            return enc, memory, seq_lengths

        enc = self.empty_candidate_code_hidden.expand(batch_size, -1)
        return enc, None, None

    def decode(self, hidden, memory_attn_mask, outputs):
        return self.decoder(hidden, memory_attn_mask, data.replace_pad_with_end(outputs))

    def decode_token(self, t, hidden, memory_attn_mask, attentions=None):
        return self.decoder.decode_token(t, hidden, memory_attn_mask, attentions)

    def sample(self, hidden, memory_attn_mask, attentions=None):
        return self.decoder.sample(hidden, memory_attn_mask, attentions=attentions)
