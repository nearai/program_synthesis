import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class TransformerEmbedding(nn.Module):
    """Embeddings matrix mixed with positional encoding for sequences."""

    def __init__(self, vocab_size, num_units, use_positional_encoding):
        super(TransformerEmbedding, self).__init__()
        self.num_units = num_units
        self.use_positional_encoding = use_positional_encoding
        self.embed = nn.Embedding(vocab_size, num_units)

    def _sin_cos_enc(self, from_length, to_length, embedding_size):
        position_enc = np.array(
            [[
                pos / np.power(10000, 2 * i / embedding_size)
                for i in range(embedding_size)
            ] for pos in range(from_length, to_length)],
            dtype=np.float32)

        # put sinusodial on even position
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        # put cosine on odd position
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        result = Parameter(torch.from_numpy(position_enc))
        if next(self.embed.parameters()).is_cuda:
            result = result.cuda()
        return result

    def forward(self, ids):
        embed = self.embed(ids)
        if self.use_positional_encoding:
            embed += self._sin_cos_enc(0, embed.size(1), self.num_units)
        return embed

    def sample(self, ids, position):
        embed = self.embed(ids)
        if self.use_positional_encoding:
            embed += self._sin_cos_enc(position, position + 1, self.num_units)
        return embed
