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

import torchfold

from program_synthesis.common.tools import saver

from program_synthesis.karel.dataset import data
from program_synthesis.karel.dataset import executor


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

    def process_infer_results(self, batch, inference_results):
        raise NotImplementedError

    def debug(self, batch):
        raise NotImplementedError

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

    def batch_processor(self, for_eval):
        '''Returns a function used to process batched data for this class.'''
        def default_processor(batch):
            return batch
        return default_processor


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

    def reset_vocab(self):
        self.last_vocab = data.PlaceholderVocab(
            self.vocab, self.args.num_placeholders)
        return self.last_vocab

    def _try_sequences(self, vocab, sequences, batch, beam_size):
        result = [[] for _ in range(len(batch))]
        counters = [0 for _ in range(len(batch))]
        candidates = [[] for _ in range(len(batch))]
        max_eval_trials = self.args.max_eval_trials or beam_size
        for batch_id, outputs in enumerate(sequences):
            example = batch[batch_id]
            #print("===", example.code_tree)
            candidates[batch_id] = [[vocab.itos(idx) for idx in ids]
                                    for ids in outputs]
            for code in candidates[batch_id][:max_eval_trials]:
                counters[batch_id] += 1
                stats = executor.evaluate_code(
                    code, example.schema.args, example.input_tests, self.executor.execute)
                ok = (stats['correct'] == stats['total'])
                #print(code, stats)
                if ok:
                    result[batch_id] = code
                    break
        return [InferenceResult(code_sequence=seq, info={'trees_checked': c, 'candidates': cand})
                for seq, c, cand in zip(result, counters, candidates)]
