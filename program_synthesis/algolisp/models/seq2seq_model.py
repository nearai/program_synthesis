from __future__ import absolute_import

import os
import sys
import collections

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torchfold

from program_synthesis.common.models import beam_search
from program_synthesis.common.modules import decoders

from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.dataset import executor

from program_synthesis.algolisp.models import prepare_spec
from program_synthesis.algolisp.models.base import InferenceResult, MaskedMemory, get_attn_mask
from program_synthesis.algolisp.models.seq2code_model import Seq2CodeModel
from program_synthesis.algolisp.models.modules import seq2seq


class Seq2SeqModel(Seq2CodeModel):

    def __init__(self, args):
        self.vocab = data.load_vocabs(args.word_vocab, args.code_vocab, args.num_placeholders)
        self.model = seq2seq.Seq2SeqAttn(
            self.vocab.word_vocab_size, self.vocab.code_vocab_size, args)
        self._executor = None
        super(Seq2SeqModel, self).__init__(args)

    @property
    def executor(self):
        if self._executor is None:
            self._executor = executor.get_executor(self.args)()
        return self._executor

    def encode(self, vocab, batch, volatile):
        inputs = prepare_spec.encode_text(
            vocab.wordtoi, batch, self.args.cuda, volatile)
        hidden, memory = self.model.encode_text(inputs)
        memory, seq_lengths, hidden = memory.pad(batch_first=True,
                                                 others_to_unsort=[hidden])
        attn_mask = get_attn_mask(seq_lengths, self.args.cuda) if seq_lengths else None

        return hidden, (memory, attn_mask)

    def decode(self, vocab, batch, hidden, memory, volatile):
        outputs = prepare_spec.encode_output_code_seq(
            vocab.codetoi, batch, self.args.cuda, volatile)
        logits, _ = self.model.decode(hidden, memory, outputs[:, :-1])
        return logits.view(-1, logits.size(2)), outputs[:, 1:].contiguous().view(-1)

    def compute_loss(self, batch, volatile=False):
        vocab = self.reset_vocab()
        hidden, memory = self.encode(vocab, batch, volatile)
        logits, labels = self.decode(vocab, batch, hidden, memory, volatile)
        return self.criterion(logits, labels)

    def compute_loss_(self, batch, volatile):
        return self.compute_loss(batch, volatile)

    def inference(self, batch):
        vocab = self.reset_vocab()
        hidden, memory = self.encode(vocab, batch, volatile=True)
        beam_size = self.args.max_beam_trees
        if beam_size > 1:
            sequences = beam_search.beam_search(
                len(batch),
                decoders.BeamSearchState(
                    [hidden for _ in range(self.args.num_decoder_layers)],
                    prev_output=None),
                MaskedMemory(memory[0], memory[1]),
                self.model.decoder.decode_token,
                beam_size,
                cuda=self.args.cuda,
                max_decoder_length=self.args.max_decoder_length)
            return self._try_sequences([vocab.itocode]*len(sequences), sequences, batch, beam_size)
        else:
            result_ids = self.model.sample(hidden, memory)
            return [InferenceResult(code_sequence=[vocab.itocode(idx) for idx in ids]) for ids in result_ids]
