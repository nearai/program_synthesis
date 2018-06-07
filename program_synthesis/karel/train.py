import argparse
import collections
import copy
import math
import random
import sys
import os
import json

import torch
import tqdm

from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models
from program_synthesis import tools
from program_synthesis.algolisp.tools import timer


def loop_iterable(x):
    '''Repeatedly yield items from `x`.'''
    while True:
        for item in x:
            yield item


def trigger(step, frac):
    '''Returns True on integer step values `frac` fraction of the time.'''
    return math.floor(step * frac) != math.floor((step - 1) * frac)


def trigger_random(step, frac):
    return random.random() <= frac


def get_sampler(train_data, args):
    sampler = train_data
    # TODO: fix this to work with the torch DataLoader.
    if args.dataset_bucket:
        buckets = [10, 50] + [x for x in range(100, 2000, 100)]
        def map_to_bucket(example):
            for size in buckets:
                if len(example.code_sequence) <= size:
                    return size
            return buckets[-1]
        def adaptive_size(res):
            max_size = max([len(example.code_sequence) for example in res])
            return max_size * len(res) > 400 * train_data.batch_size
        sampler = dataset.dataset.BucketizedSampler(sampler, buckets, map_to_bucket, adaptive_size)
    return sampler


def train_start(args):
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    dataset.set_vocab(args)
    m = models.get_model(args)
    m.model.train()
    train_data = dataset.get_train_dataset(args, m, for_eval=False)
    dev_data = dataset.get_eval_dataset(args, m)
    dev_data.shuffle = True
    sampler = get_sampler(train_data, args)
    tools.save_args(args)
    return train_data, dev_data, m, sampler


def train(args):
    print("Training:")
    train_data, dev_data, m, sampler = train_start(args)
    reporter = tools.Reporter(
        log_interval=args.log_interval,
        logdir=args.model_dir, smooth_interval=args.log_interval)
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(sampler):
            res = m.train(batch)
            reporter.record(m.last_step, **res)
            reporter.report()
            if m.last_step % args.eval_every_n == 0:
                m.model.eval()
                stats = collections.defaultdict(int)
                for dev_idx, dev_batch in tqdm.tqdm(enumerate(dev_data)):
                    batch_res = m.eval(dev_batch)
                    for k, v in batch_res.items():
                        if isinstance(v, dict):
                            if k not in stats:
                                stats[k] = v
                            else:
                                for i1, i2 in v.items():
                                    stats[k][i1] += i2
                        else:
                            stats[k] += v
                    if args.eval_n_steps and dev_idx > args.eval_n_steps:
                        break
                if 'correct' in stats:
                    stats['accuracy'] = stats['correct']
                    del stats['correct']
                total = float(stats['total'])
                del stats['total']
                if 'correct_per_key' in stats:
                    for key in stats['correct_per_key']:
                        stats['accuracy/key=%s' % key] = (
                            float(stats['correct_per_key'][key]) /
                            stats['total_per_key'][key] * total)
                    del stats['correct_per_key']
                    del stats['total_per_key']
                for k in stats:
                    stats[k] /= total
                print("Step {} stats: ".format(m.last_step) + ", ".join(
                    "{} = {}".format(k, v) for k, v in stats.items()))
                reporter.record(m.last_step,
                                **{'{}/dev'.format(k): v
                                   for k, v in stats.items()})
                m.model.train()


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Training Text2Code', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)
