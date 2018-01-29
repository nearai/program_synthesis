import collections
import math
import sys
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from pytorch_tools import torchfold

from datasets import data
from tools import saver


class MaskedMemory(collections.namedtuple('MaskedMemory', ['memory',
    'attn_mask'])):

    def expand_by_beam(self, beam_size):
        return MaskedMemory(*(v.unsqueeze(1).repeat(1, beam_size, *([1] * (
            v.dim() - 1))).view(-1, *v.shape[1:]) for v in self))


def get_attn_mask(seq_lengths, cuda):
    max_length, batch_size = max(seq_lengths), len(seq_lengths)
    ranges = torch.arange(
        0, max_length,
        out=torch.LongTensor()).unsqueeze(0).expand(batch_size, -1)
    attn_mask = (ranges >= torch.LongTensor(seq_lengths).unsqueeze(1))
    if cuda:
        attn_mask = attn_mask.cuda()
    return attn_mask


class InferenceResult(object):

    def __init__(self, code_tree=None, code_sequence=None, info=None):
        self.code_tree = code_tree
        self.code_sequence = code_sequence
        self.info = info

    def to_dict(self):
        return {
            'info': self.info,
            'code_tree': self.code_tree if self.code_tree else [],
            'code_sequence': self.code_sequence,
        }


class BaseModel(object):

    def __init__(self, args):
        self.args = args
        self.model_dir = args.model_dir
        self.save_every_n = args.save_every_n
        self.debug_every_n = args.debug_every_n

        self.saver = saver.Saver(self.model, self.optimizer, args.keep_every_n)
        self.last_step = self.saver.restore(
            self.model_dir, map_to_cpu=args.restore_map_to_cpu,
            step=getattr(args, 'step', None))
        if self.last_step == 0 and args.pretrained:
            for kind_path in args.pretrained.split(','):
                kind, path = kind_path.split(':')
                self.load_pretrained(kind, path)

    def load_pretrained(self, kind, path):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def inference(self, batch):
        raise NotImplementedError

    def format_code(self, tree, seq, lang):
        if tree is not None:
            return data.format_code(tree, lang)
        else:
            try:
                tree, complete = data.unflatten_code(seq, lang)
            except:
                complete = False
            if complete:
                return data.format_code(tree, lang)
            return ' '.join(seq)

    def debug(self, batch):
        lang = batch[0].language
        print("Text:   %s" % ' '.join(batch[0].text))
        if batch[0].funcs:
            funcs = '\n'.join(
                ['\t%s: %s' % (code_func.name, ' '.join(code_func.code_sequence))
                 for code_func in batch[0].funcs])
            print(funcs)
        print("Schema: %s" % ', '.join(batch[0].schema.args))

        if batch[0].candidate_code_sequence is not None:
            print("Cand:   %s" % self.format_code_seq(
                batch[0].candidate_code_sequence, lang))

        print("Code:  %s" % self.format_code(
            batch[0].code_tree, batch[0].code_sequence, lang))

        res = self.inference([batch[0]])
        print("Res:   %s" % self.format_code(
            res[0].code_tree, res[0].code_sequence, lang))

        if res[0].info:
            print("Info:   %s" % res[0].info)

    def train(self, batch):
        self.update_lr()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        if self.args.gradient_clip is not None:
            nn.utils.clip_grad_norm(self.model.parameters(),
                                    self.args.gradient_clip)
        self.optimizer.step()
        self.last_step += 1
        if self.debug_every_n > 0 and self.last_step % self.debug_every_n == 0:
            self.debug(batch)
        if self.last_step % self.save_every_n == 0:
            self.saver.save(self.model_dir, self.last_step)
        return {'loss': loss.data[0]}

    def eval(self, batch):
        results = self.inference(batch)
        correct = 0
        for example, res in zip(batch, results):
            if example.code_sequence == res.code_sequence or example.code_tree == res.code_tree:
                correct += 1
        return {'correct': correct, 'total': len(batch)}

    def update_lr(self):
        args = self.args
        if args.lr_decay_steps is None or args.lr_decay_rate is None:
            return

        lr = args.lr * args.lr_decay_rate ** (self.last_step //
                                              args.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class BaseCodeModel(BaseModel):

    def __init__(self, args):
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            raise ValueError(args.optimizer)

        super(BaseCodeModel, self).__init__(args)

        if args.cuda:
            self.model.cuda()
        print(self.model)

    def reset_vocab(self):
        self.last_vocab = data.PlaceholderVocab(
            self.vocab, self.args.num_placeholders)
        return self.last_vocab
