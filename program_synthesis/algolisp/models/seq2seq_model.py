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
from program_synthesis.common.modules import seq2seq

from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.dataset import executor

from program_synthesis.algolisp.models import prepare_spec
from program_synthesis.algolisp.models.base import InferenceResult, MaskedMemory, get_attn_mask
from program_synthesis.algolisp.models.seq2code_model import Seq2CodeModel
from program_synthesis.algolisp.models.modules import encoders


class Spec2Seq(seq2seq.Sequence2Sequence):

    def __init__(self, input_vocab_size, output_vocab_size, args):
        super(Spec2Seq, self).__init__(
            input_vocab_size, output_vocab_size, args, encoder_cls=encoders.SpecEncoder
        )

    def encode_text(self, inputs):
        # inputs: PackedSequencePlus
        return self.encoder.text_encoder(inputs)

    def encode_io(self, input_keys, inputs, arg_nums, outputs):
        input_keys_embed = self.decoder.embed(input_keys)
        return self.encoder.io_encoder(input_keys_embed, inputs, arg_nums, outputs)

    def encode_code(self, code_seqs):
        # code_seqs: PackedSequencePlus
        return self.encoder.code_encoder(code_seqs.apply(self.decoder.embed))

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


class Seq2SeqModel(Seq2CodeModel):

    def __init__(self, args):
        self.vocab = data.load_vocabs(args.word_vocab, args.code_vocab, args.num_placeholders, getattr(args, 'vocab_mapping', True))
        self.model = Spec2Seq(
            self.vocab.word_vocab_size + args.num_placeholders, self.vocab.code_vocab_size + args.num_placeholders, args)
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
        logits = self.model.decode(hidden, memory, data.replace_pad_with_end(outputs[:, :-1]))
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
            return [InferenceResult(code_sequence=[vocab.itocode(idx.item()) for idx in ids]) for ids in result_ids]
