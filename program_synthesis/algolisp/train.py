import collections
import sys
import os
import multiprocessing as mp

import torch

from program_synthesis.common.tools import reporter as reporter_lib
from program_synthesis.common.tools import saver

from program_synthesis.algolisp import arguments
from program_synthesis.algolisp import models
from program_synthesis.algolisp.dataset import dataset
from program_synthesis.algolisp.dataset import evaluation


def train_start(args):
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    train_data, dev_data = dataset.get_dataset(args)
    m = models.get_model(args)
    m.model.train()
    saver.save_args(args)
    return train_data, dev_data, m


def train(args):
    print("Training:")
    train_data, dev_data, m = train_start(args)
    dev_data = [('Dev', dev_data)] if not isinstance(dev_data, list) else dev_data
    t = reporter_lib.Tee(os.path.join(args.model_dir, "output.log"))
    reporter = reporter_lib.Reporter(
        log_interval=args.log_interval, logdir=args.model_dir,
        smooth_interval=args.log_interval)
    dev_reporters = {}
    for label, _ in dev_data:
        dev_reporters[label] = reporter_lib.Reporter(
            log_interval=args.eval_every_n, logdir=os.path.join(args.model_dir, label),
            smooth_interval=1)

    m.worker_pool = mp.Pool(mp.cpu_count())
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(train_data):
            res = m.train(batch)
            reporter.record(m.last_step, **res)
            reporter.report()
            if m.last_step % args.eval_every_n == 0:
                m.model.eval()
                for label, dd in dev_data:
                    all_stats = []
                    for stats_id, stats in enumerate(evaluation.run_inference(dd, m, m._executor)):
                        if stats_id > args.eval_n_examples:
                            break
                        all_stats.append(stats)
                    metrics = evaluation.compute_metrics(all_stats)
                    dev_reporters[label].record(m.last_step, **metrics)
                    dev_reporters[label]._writer.flush()
                    # use report from dev_reproters instead of manually
                    # reporting here.
                    out_metrics = ["%s=%.4f" % (k, v) for k, v in metrics.items()]
                    print("Dev %s metrics: %s" % (label, ', '.join(out_metrics)))
                m.model.train()
    t.close()


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Training AlgoLisp', 'train')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)
