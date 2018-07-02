from __future__ import absolute_import

import os
import sys
import collections

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from program_synthesis.algolisp.dataset import data

from program_synthesis.algolisp.models import prepare_spec
from program_synthesis.algolisp.models.base import BaseCodeModel, InferenceResult, MaskedMemory, get_attn_mask


class Seq2CodeModel(BaseCodeModel):

    def encode_text(self, stoi, batch):
        inputs = prepare_spec.encode_schema_text(
            stoi, batch, self.args.cuda)
        hidden, memory = self.model.encode_text(inputs)
        memory, seq_lengths, hidden = memory.pad(batch_first=True,
                others_to_unsort=[hidden])

        return hidden, memory, seq_lengths

    def encode_io(self, stoi, batch):
        input_keys, inputs, arg_nums, outputs = prepare_spec.encode_io(
            stoi, batch, self.args.cuda)
        task_enc = self.model.encode_io(input_keys, inputs, arg_nums, outputs)
        return task_enc, task_enc.unsqueeze(1)

    def encode_code(self, stoi, batch):
        batch_ids, (code_seqs, unsort_idx) = prepare_spec.encode_candidate_code_seq(
            stoi, batch, self.args.cuda)

        code_info = None
        if code_seqs:
            hidden, memory = self.model.encode_code(code_seqs)
            memory, seq_lengths, hidden = memory.pad(batch_first=True,
                    others_to_unsort=[hidden])
            code_info = (hidden, memory, seq_lengths)

        return self.model.extend_tensors(code_info, len(batch), batch_ids)

    def encode(self, vocab, batch):
        text_task_enc, text_memory, text_lengths = None, None,  None
        io_task_enc = None
        code_enc, code_memory, code_lengths = None, None, None

        if self.args.read_text:
            text_task_enc, text_memory, text_lengths = self.encode_text(
                vocab.wordtoi, batch)
        if self.args.read_io:
            io_task_enc, _ = self.encode_io(vocab.codetoi, batch)

        hidden, memory, seq_lengths = self.model.encoder(
            text_task_enc, text_memory, text_lengths, io_task_enc, code_enc,
            code_memory, code_lengths)

        attn_mask = get_attn_mask(seq_lengths, self.args.cuda) if seq_lengths else None
        return hidden, MaskedMemory(memory, attn_mask)
