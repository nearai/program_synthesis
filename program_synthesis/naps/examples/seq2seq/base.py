import collections

import torch
import torch.nn as nn
from torch import optim


from program_synthesis.naps.examples.seq2seq import data, executor, bleu
from program_synthesis.common.tools import saver


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

        self.saver = saver.Saver(self.model, self.optimizer, args.keep_every_n)
        self.last_step = self.saver.restore(
            self.model_dir, map_to_cpu=args.restore_map_to_cpu,
            step=getattr(args, 'step', None))
        
        # Init working params.
        self.last_loss = None
        self.lr = args.lr
        # Multiprocessing worker pool for executing CPU-intense stuff.
        self.worker_pool = None

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
        if args.lr_decay_steps is None or args.lr_decay_rate is None:
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
            code, example['search_tests'], executor_)
        ok = (stats['tests-executed'] == stats['tests-passed'])
        if ok:
            return code
    return codes[0]
