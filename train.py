import argparse
import copy
import math
import random
import sys
import os
import json

import torch

# Add current directory to sys PATH.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

import arguments
import datasets
import models
import tools
from tools import timer


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
        sampler = datasets.dataset.BucketizedSampler(sampler, buckets, map_to_bucket, adaptive_size)
    return sampler


def train_start(args):
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    datasets.set_vocab(args)
    m = models.get_model(args)
    m.model.train()
    train_data, dev_data = datasets.get_dataset(args, m)
    dev_data.shuffle = True
    sampler = get_sampler(train_data, args)
    tools.save_args(args)
    return train_data, dev_data, m, sampler


def train(args):
    print("Training:")
    train_data, dev_data, m, sampler = train_start(args)
    reporter = tools.Reporter(log_interval=args.log_interval, logdir=args.model_dir)
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(sampler):
            res = m.train(batch)
            reporter.record(m.last_step, **res)
            reporter.report()
            if m.last_step % args.eval_every_n == 0:
                m.model.eval()
                stats = {'correct': 0, 'total': 0}
                for dev_idx, dev_batch in enumerate(dev_data):
                    batch_res = m.eval(dev_batch)
                    stats['correct'] += batch_res['correct']
                    stats['total'] += batch_res['total']
                    if dev_idx > args.eval_n_steps:
                        break
                accuracy = float(stats['correct']) / stats['total']
                print("Dev accuracy: %.5f" % accuracy)
                reporter.record(m.last_step, **{'accuracy/dev': accuracy})
                m.model.train()


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Training Text2Code', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)
