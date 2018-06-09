from __future__ import absolute_import

import collections
import math
import multiprocessing
import sys
import os
import time

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from program_synthesis.common.tools import saver
from program_synthesis.algolisp.tools import bleu

from program_synthesis.algolisp.dataset import data
from program_synthesis.algolisp.dataset import executor
from program_synthesis.algolisp.dataset import executor


class MaskedMemory(collections.namedtuple('MaskedMemory', ['memory',
    'attn_mask'])):

    def expand_by_beam(self, beam_size):
        return MaskedMemory(*(v.unsqueeze(1).repeat(1, beam_size, *([1] * (
            v.dim() - 1))).view(-1, *v.shape[1:]) if v is not None else None for v in self))


def get_attn_mask(seq_lengths, cuda):
    max_length, batch_size = max(seq_lengths), len(seq_lengths)
    t_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    ranges = torch.arange(
        0, max_length,
        out=t_type()).unsqueeze(0).expand(batch_size, -1)
    attn_mask = (ranges >= t_type(seq_lengths).unsqueeze(1))
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
        
        # Init working params.
        self.last_loss = None
        self.lr = args.lr
        # Multiprocessing worker pool for executing CPU-intense stuff.
        self.worker_pool = None

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
        if lang != 'uidsl':
            print("Text:   %s" % ' '.join(batch[0].text))
            if hasattr(batch[0], 'funcs') and batch[0].funcs:
                funcs = '\n'.join(
                    ['\t%s: %s' % (code_func.name, ' '.join(code_func.code_sequence))
                     for code_func in batch[0].funcs])
                print(funcs)
            print("Schema: %s" % ', '.join(batch[0].schema.args))

            if hasattr(batch[0], 'candidate_code_sequence') and batch[0].candidate_code_sequence is not None:
                print("Cand:   %s" % self.format_code_seq(
                    batch[0].candidate_code_sequence, lang))

        res = self.inference([batch[0]])

        if res[0].code_tree is not None:
            print("Code:  %s" % self.format_code(batch[0].code_tree, None, lang))
            print("Res:   %s" % self.format_code(res[0].code_tree, None, lang))
        else:
            print("Code:  %s" % self.format_code(None, batch[0].code_sequence, lang))
            print("Res:   %s" % self.format_code(None, res[0].code_sequence, lang))

        if res[0].info:
            print("Info:   %s" % res[0].info)

        if hasattr(self, 'last_vocab') and hasattr(self.last_vocab, 'unks'):
            unks = sorted(self.last_vocab.unks.items(), key=lambda x: -x[1])
            print("Unks:  %d (%s)" % (len(unks), unks[:5]))

    def train(self, batch):
        self.update_lr()
        self.optimizer.zero_grad()
        try:
            loss = self.compute_loss(batch)
        except RuntimeError:
            raise
        if self.last_loss is not None and self.last_step > 1000 and loss.data[0] > self.last_loss * 3:
            print("Loss exploded: %f to %f. Skipping batch." % 
                (self.last_loss, loss.data[0]))
            return {'loss': self.last_loss, 'lr': self.lr}

        loss.backward()
        if self.args.clip is not None:
            torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                          self.args.clip)
        self.optimizer.step()
        self.last_step += 1
        if self.debug_every_n > 0 and self.last_step % self.debug_every_n == 0:
            self.debug(batch)
        if self.last_step % self.save_every_n == 0:
            self.saver.save(self.model_dir, self.last_step)
        self.last_loss = loss.data[0]
        return {'loss': loss.data[0], 'lr': self.lr}

    def update_eval_metrics(self, example, res, result):
        if (example.code_sequence == res.code_sequence or 
            (hasattr(example, 'code_tree') and example.code_tree == res.code_tree)):
            result['correct'] += 1.0
        else:
            result['correct'] += 0.0
        if example.code_sequence is not None and res.code_sequence is not None:
            try:
                result['bleu'] += bleu.compute_bleu([[example.code_sequence]], [res.code_sequence])
            except ZeroDivisionError:
                result['bleu'] += 0.0

    def eval(self, batch, metrics=None):
        results = self.inference(batch)
        if metrics is None:
            metrics = collections.defaultdict(float)
        metrics['total'] += len(batch)
        # XXX: adding this to handle volatile for loss computation.
        if hasattr(self, 'compute_loss_'):
            train_loss = self.compute_loss_(batch, volatile=True)
        else:
            train_loss = self.compute_loss(batch)
        metrics['loss'] += train_loss.data[0] * len(batch)
        for example, res in zip(batch, results):
            self.update_eval_metrics(example, res, metrics)
        return metrics

    def update_lr(self):
        args = self.args
        if  args.lr_decay_steps is None or args.lr_decay_rate is None:
            return

        self.lr = args.lr * args.lr_decay_rate ** (self.last_step //
                args.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


class BaseCodeModel(BaseModel):

    def __init__(self, args):
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr,
                weight_decay=args.optimizer_weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            raise ValueError(args.optimizer)

        super(BaseCodeModel, self).__init__(args)
        if args.cuda:
            self.model.cuda()
        print(self.model)

    def reset_vocab(self):
        if isinstance(self.vocab, data.WordCodeVocab):
            self.vocab.reset()
            self.last_vocab = self.vocab
            # self.last_vocab = data.WordCodeVocab(
            #     word=data.PlaceholderVocab(self.vocab.word, self.args.num_placeholders),
            #     code=data.PlaceholderVocab(
            #         self.vocab.code, self.args.num_placeholders)
            # )
        else:
            self.last_vocab = data.PlaceholderVocab(
                self.vocab, self.args.num_placeholders)
        return self.last_vocab

    def _try_sequences(self, itos_funcs, sequences, batch, beam_size):
        result = [[] for _ in range(len(batch))]
        counters = [0 for _ in range(len(batch))]
        candidates = [
            [[itos(idx) for idx in ids] for ids in outputs] for outputs, itos in zip(sequences, itos_funcs)
        ]
        max_eval_trials = self.args.max_eval_trials
        if max_eval_trials == 0:
            return [
                InferenceResult(code_sequence=seq[0], info={'candidates': seq}) 
                for seq in candidates]
        map_func = self.worker_pool.imap if self.worker_pool is not None else map
        for batch_id, best_code in enumerate(map_func(get_best_code, (
                (example, candidate[:max_eval_trials], self.executor)
                for example, candidate in zip(batch, candidates)))):
            result[batch_id] = best_code
        return [InferenceResult(code_sequence=seq, info={'trees_checked': c, 'candidates': cand})
                for seq, c, cand in zip(result, counters, candidates)]


def get_best_code(args):
    example, codes, executor_ = args
    for code in codes:
        stats = executor.evaluate_code(
            code, example.schema.args, example.input_tests, executor_)
        ok = (stats['tests-executed'] == stats['tests-passed'])
        if ok:
            return code
    return codes[0]
