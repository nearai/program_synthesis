import numpy as np

import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .layer_norm import LayerNorm


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert attn_mask.size() == attn.size(), \
            'Attention mask shape {} mismatch ' \
            'with Attention logit tensor shape ' \
            '{}.'.format(attn_mask.size(), attn.size())

        attn.data.masked_fill_(attn_mask, -float('inf'))


class DotProductAttention(nn.Module):

    def __init__(self, num_units, num_mem_units, num_heads):
        super(DotProductAttention, self).__init__()
        self.linear_ins = [
            nn.Linear(num_units, num_mem_units, bias=False) for _ in range(num_heads)]
        self.linear_outs = [nn.Linear(
            num_mem_units + 2 * num_units, num_units, bias=False) for _ in range(num_heads)]

        for i, x in enumerate(self.linear_ins + self.linear_outs):
            setattr(self, 'param_%s' % i, x)

        self.num_heads = num_heads

    def forward(self, query, context, attn_mask=None):
        """Apply attention.

        query: batch x dim
        context: batch x length x dim
        """
        input_ = query
        for i in range(self.num_heads):
            query_proj = self.linear_ins[i](
                input_).unsqueeze(2)  # batch x dim x 1
            attn = torch.bmm(context, query_proj).squeeze(2)  # batch x length

            maybe_mask(attn, attn_mask)

            attn = F.softmax(attn, dim=1)
            wc = torch.bmm(attn.unsqueeze(1), context).squeeze(1)  # batch x dim
            wc = torch.cat([wc, input_, query], 1)  # batch x 2dim
            wc = self.linear_outs[i](wc)
            wc = torch.tanh(wc)
            input_ = wc
        return wc, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dim, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(dim, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        maybe_mask(attn, attn_mask)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class RepeatLinear(nn.Module):

    def __init__(self, repeat, feature_dim, dim):
        super(RepeatLinear, self).__init__()
        self.repeat = repeat
        self.layer = nn.Parameter(torch.FloatTensor(repeat, feature_dim, dim))
        self.output_dim = dim
        init.xavier_normal(self.layer)

    def forward(self, x):
        _, dim1, dim2 = x.size()
        if self.repeat > 1:
            out = x.repeat(self.repeat, 1, 1).view(self.repeat, -1, dim2)
        else:
            out = x.view(1, -1, dim2)
        return torch.bmm(out, self.layer).view(-1, dim1, self.output_dim)


class MultiHeadAttention(nn.Module):

    def __init__(
            self, num_heads, num_units, query_dim, key_dim, value_dim,
            dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_units = num_units
        assert query_dim == key_dim

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_layer = RepeatLinear(num_heads, num_units, query_dim)
        self.key_layer = RepeatLinear(num_heads, num_units, key_dim)
        self.value_layer = RepeatLinear(num_heads, num_units, value_dim)
        self.attention = ScaledDotProductAttention(num_units)

        self.proj = nn.Linear(num_heads * value_dim, num_units)
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = LayerNorm(num_units)

    def forward(self, query, keys, values, attn_mask=None):
        # query shape: batch x num queries x num units
        # keys shape: batch x num kv x num units
        # values shape: batch x num kv x num units

        # batch * heads x num queries x query_dim
        Q = self.query_layer(query)
        # batch * heads x num kv x key_dim (= query_dim)
        K = self.key_layer(keys)
        # batch * heads x num kv x value_dim
        V = self.value_layer(values)

        # outputs: batch * heads x num queries x value_dim
        # attns: batch * heads x num queries x num kv
        outputs, attns = self.attention(
            Q, K, V, attn_mask=attn_mask.repeat(self.num_heads, 1, 1) if attn_mask is not None else None)

        # TODO: transpose or unfold?
        bsz = query.size(0)
        # batch x num queries x num_heads * value_dim
        outputs = torch.cat(torch.split(outputs, bsz, dim=0), dim=-1)

        # batch x num queries x num_units
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs + query), attns


class SimpleMultiHeadAttention(MultiHeadAttention):
    def __init__(self, num_heads, num_units, dropout_p=0.1):
        assert num_units % num_heads == 0
        dim = num_units / num_heads
        super(SimpleMultiHeadAttention, self).__init__(
            num_heads, num_units, dim, dim, dim, dropout_p)

    def forward(self, query, values, attn_mask=None):
        if query.dim() == 2:
            query = query.unsqueeze(1)

        outputs, attns = super(SimpleMultiHeadAttention, self).forward(
            query, values, values, attn_mask)

        if query.dim() == 2:
            outputs = outputs.squeeze(1)

        return outputs, attns


class SimpleSDPAttention(ScaledDotProductAttention):
    def __init__(self, query_dim, values_dim, dropout_p=0.0):
        super(SimpleSDPAttention, self).__init__(values_dim, dropout_p)
        self.query_proj = nn.Linear(query_dim, values_dim)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query dim
        # values shape: batch x num values x values dim
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
        output, attn = super(SimpleSDPAttention, self).forward(
            self.query_proj(query).unsqueeze(1), values, values, attn_mask)

        output = output.squeeze(1)
        return output, attn
