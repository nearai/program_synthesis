import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from program_synthesis.common.modules.layer_norm import LayerNorm
from program_synthesis.common.modules.embedding import TransformerEmbedding
from program_synthesis.common.modules.attention import MultiHeadAttention


class Stack(nn.Sequential):

    def __init__(self, *args):
        super(Stack, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, (tuple, list)):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PositionWiseFFN(nn.Module):

    def __init__(self, dim, num_units, dropout_p=0.1):
        super(PositionWiseFFN, self).__init__()
        layers = []
        for idx, unit in enumerate(num_units):
            # layers.append(nn.Conv1d(dim, unit, 1))
            layers.append(nn.Linear(dim, unit))
            if idx < len(num_units) - 1:
                layers.append(nn.ReLU())
            dim = unit
        layers.append(nn.Dropout(dropout_p))
        self.layers = nn.Sequential(*layers)
        self.layer_norm = LayerNorm(dim)

    def forward(self, x):
        return self.layer_norm(self.layers(x) + x)


class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_units, ffn_units, num_att_heads):
        super(TransformerEncoder, self).__init__()
        layers = []
        for unit in num_units:
            layers.append(Stack(
                MultiHeadAttention(num_att_heads, unit, dim, dim, dim),
                PositionWiseFFN(unit, ffn_units)))
            dim = unit
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        net_inputs = inputs
        for layer in self.layers:
            res1, _ = layer[0](net_inputs, net_inputs, net_inputs)
            net_inputs = layer[1](res1)
        return net_inputs


def get_attn_padding_mask(seq_q, seq_k, pad_token=1):
    """Indicate the padding-related part to mask"""
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_token).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    """Get an attention mask to avoid using the subsequent info."""
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).clone()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class TransformerDecoder(nn.Module):

    def __init__(self, query_dim, key_dim, num_units, ffn_units, num_att_heads):
        super(TransformerDecoder, self).__init__()
        layers = []
        for unit in num_units:
            layers.append(Stack(
                MultiHeadAttention(8, unit, query_dim, query_dim, query_dim),
                MultiHeadAttention(8, unit, key_dim, unit, unit),
                PositionWiseFFN(unit, ffn_units)
            ))
            dim = unit
        self.layers = nn.ModuleList(layers)

    def forward(self, query, keys, attn_mask):
        for idx, layer in enumerate(self.layers):
            res1, att = layer[0](query, query, query, attn_mask=attn_mask)
            res2, _ = layer[1](res1, keys, keys)
            query = layer[2](res2)
        return query

    def decode_token(self, query, prev_state, keys):
        state = []
        for idx, layer in enumerate(self.layers):
            if prev_state is not None:
                query_ = torch.cat([prev_state[idx], query], 1)
            else:
                query_ = query                
            res1, att = layer[0](query, query_, query_)
            state.append(query_)
            res2, _ = layer[1](res1, keys, keys)
            query = layer[2](res2)
        return query, torch.stack(state)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed, vocab_size, dropout_p, query_dim, key_dim, num_units, ffn_units, num_att_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.embed = embed
        self.dropout_p = dropout_p
        self.decoder = TransformerDecoder(query_dim, key_dim, num_units, ffn_units, num_att_heads)
        self.output_layer = nn.Linear(ffn_units[-1], vocab_size)

    def forward(self, encoder_result, dec_input):
        dec_embed = self.embed(dec_input)
        dec_embed = F.dropout(dec_embed, self.dropout_p)
        attn_mask = get_attn_subsequent_mask(dec_input)

        decoder_result = self.decoder(dec_embed, encoder_result, attn_mask)
        decoder_result = F.dropout(decoder_result, self.dropout_p)

        output = self.output_layer(decoder_result)
        softmax = F.softmax(output, dim=-1)
        return softmax, output

    def decode_token(self, encoder_result, position, dec_input, prev_state):
        dec_embed = self.embed.sample(dec_input.unsqueeze(1), position)
        dec_embed = F.dropout(dec_embed, self.dropout_p)
        decoder_result, new_state = self.decoder.decode_token(dec_embed, prev_state, encoder_result)
        decoder_result = F.dropout(decoder_result, self.dropout_p)

        output = self.output_layer(decoder_result)
        softmax = F.softmax(output, dim=-1)
        return softmax, output, new_state

    def sample(self, encoder_result, cuda, max_decoder_length):
        bsz = encoder_result.size(0)
        LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
        last_state = None
        last_input = Variable(LongTensor([0] * bsz), volatile=True)
        output = [[] for _ in range(bsz)]
        for pos in range(max_decoder_length):
            prob, zz, last_state = self.decode_token(encoder_result, pos, last_input, last_state)
            last_input = prob.squeeze(1).max(1)[1]
            for i in range(bsz):
                if len(output[i]) > 0 and output[i][-1] == 1:
                    continue
                output[i].append(last_input[i].data[0])
        return [x[:-1] if x[-1] == 1 else x for x in output]


class Transformer2Transformer(nn.Module):

    def __init__(self, input_vocab_size, output_vocab_size, args):
        super(Transformer2Transformer, self).__init__()
        self.args = args
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_units = args.num_units
        self.num_placeholders = args.num_placeholders
        self.dropout_p = args.dropout
        enc_units = [self.num_units] * self.args.num_transformer_layers
        dec_units = [self.num_units] * self.args.num_transformer_layers
        ffn_units = [self.num_units] * self.args.num_transformer_layers

        self.embed = TransformerEmbedding(
            input_vocab_size + self.num_placeholders, self.num_units, True)
        self.out_embed = TransformerEmbedding(
            output_vocab_size + self.num_placeholders, self.num_units, True)
        self.encoder = TransformerEncoder(self.num_units, enc_units, ffn_units, args.num_att_heads)
        self.decoder = TransformerDecoderLayer(
            self.out_embed, output_vocab_size + self.num_placeholders,
            self.dropout_p, self.num_units, self.num_units, dec_units, ffn_units, args.num_att_heads)

    def encode(self, enc_input):
        enc_embed = self.embed(enc_input)
        encoder_result = self.encoder(enc_embed)
        return F.dropout(encoder_result, self.dropout_p)

    def decode(self, encoder_result, dec_input):
        return self.decoder(encoder_result, dec_input)

    def forward(self, enc_input, dec_input):
        encoder_result = self.encode(enc_input)
        return self.decode(encoder_result, dec_input)

    def sample(self, enc_input):
        encoder_result = self.encode(enc_input)
        return self.decoder.sample(encoder_result, self.args.cuda, self.args.max_decoder_length)
