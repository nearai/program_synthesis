import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# Special symbols.
PAD_TOKEN = -1
START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
SPECIAL_SIZE = UNK_TOKEN + 1


class BatchPreparator(nn.Module):
    def __init__(self, args, vocab):
        super(BatchPreparator, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.max_oov_size = args.max_oov_size
        self.embed = nn.Embedding(SPECIAL_SIZE + self.vocab_size, args.num_units)
        self.max_code_length = args.max_code_length

    def _texts_to_tensor(self, texts, batch_size, oovs):
        lengths = [len(t) for t in texts]
        max_length = max(lengths)
        data = np.zeros((batch_size, max_length), dtype=np.int64)
        for i, (text, oov) in enumerate(zip(texts, oovs)):
            for j, token in enumerate(text):
                ind = self.vocab.get(token, None)
                if ind is None:
                    ind = oov.get(token, None)
                    if ind is None:
                        if len(oov) < self.max_oov_size:
                            ind = SPECIAL_SIZE + self.vocab_size + len(oov)
                            oov[token] = ind
                        else:
                            ind = UNK_TOKEN
                data[i, j] = ind
        return torch.LongTensor(data), lengths

    def _programs_to_tensor(self, programs, max_code_length, batch_size, oovs):
        lengths = [len(p) for p in programs]
        max_length = min(max(lengths), max_code_length)
        data = np.ones((batch_size, max_length), dtype=np.int64) * PAD_TOKEN
        for i, (program, oov) in enumerate(zip(programs, oovs)):
            for j, token in enumerate(program):
                if j == max_code_length:
                    break
                ind = self.vocab.get(token, None)
                if ind is None:
                    ind = oov.get(token, UNK_TOKEN)
                data[i, j] = ind
            if len(program) < max_code_length:
                data[i, len(program)] = END_TOKEN
        return torch.LongTensor(data)

    def _vocab_limits(self, oovs):
        result = np.array([SPECIAL_SIZE + self.vocab_size + len(oov) for oov in oovs], dtype=np.int64)
        return torch.LongTensor(result)

    def process_texts(self, batch):
        assert not self.embed.weight.is_cuda, "Batch preparation should be performed on CPU"
        texts = [d['text'] for d in batch]
        batch_size = len(batch)
        oovs = [dict() for _ in range(batch_size)]

        texts_data, texts_lengths = self._texts_to_tensor(texts, batch_size, oovs)

        # Encoder only gets tokens inside the vocab.
        encoder_input = texts_data.clone()
        encoder_input[encoder_input >= SPECIAL_SIZE + self.vocab_size] = UNK_TOKEN
        encoder_input = self.embed(Variable(encoder_input, volatile=not self.training))

        # Compute attentions mask.
        ranges = torch.arange(0, max(texts_lengths)).long().unsqueeze(0).expand(batch_size, -1)
        attn_mask = (ranges >= torch.LongTensor(texts_lengths).unsqueeze(1))

        # Enumerate text tokens for the pointer mechanism.
        indices = torch.arange(batch_size).long() * (SPECIAL_SIZE + self.vocab_size + self.max_oov_size)
        pointer_texts = Variable(texts_data + indices.unsqueeze(1), volatile=not self.training)

        return pointer_texts, texts_lengths, encoder_input, attn_mask, oovs

    def process_golden_programs(self, batch, oovs):
        assert not self.embed.weight.is_cuda, "Batch preparation should be performed on CPU"
        programs = [d['code_sequence'] for d in batch]
        max_program_len = min(max(len(p) for p in programs), self.max_code_length)
        batch_size = len(batch)

        programs_data = self._programs_to_tensor(programs, self.max_code_length, batch_size, oovs)

        # Forced output used for the supervision during training.
        forced_output = programs_data.clone()
        forced_output[forced_output >= SPECIAL_SIZE + self.vocab_size] = UNK_TOKEN
        # Inferences produced from the pad token are not used in loss function so we can replace them with anything
        # valid for the embedding.
        forced_output = forced_output.clamp(min=0)
        forced_output = self.embed(Variable(forced_output, volatile=not self.training))

        return Variable(programs_data, volatile=not self.training), forced_output, max_program_len

    def get_start_tokens(self, batch_size):
        res = Variable(torch.LongTensor(np.array([START_TOKEN]*batch_size, dtype=np.int64)),
                       volatile=not self.training)
        return self.embed(res)
