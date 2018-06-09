import time
import random
import json
import pprint
import multiprocessing as mp

import torch
import tqdm

from program_synthesis.common.tools import saver

from program_synthesis.algolisp import arguments
from program_synthesis.algolisp import models
from program_synthesis.algolisp.dataset import dataset
from program_synthesis.algolisp.dataset import evaluation
from program_synthesis.algolisp.dataset import executor


class EvalReport(object):

    def __init__(self, tag, all_stats):
        self.tag = tag
        self.all_stats = all_stats

    def save(self):
        timestamp = int(time.time())
        report_path = 'reports/report-%s-%s.json' % (self.tag, timestamp)
        with open(report_path, 'w') as f:
            metrics = evaluation.compute_metrics(self.all_stats)
            f.write(json.dumps(metrics) + "\n")
            for stats in self.all_stats:
                f.write(json.dumps(stats) + "\n")

    def show_example(self, stats):
        example = stats['example']
        res = stats['res']
        golden_lines, inferred_lines = None, None
        golden_lines = pprint.pformat(example['code_sequence'], width=100)
        inferred_lines = pprint.pformat(res['code_sequence'], width=100)
        print("STATS: %s" % {k: v for k, v in stats.items() if k not in ('example', 'res')})
        print("GOLDEN:")
        print(golden_lines)
        print("INFERENCE:")
        print(inferred_lines)

    def display(self, examples_to_show=0):
        indices = list(range(len(self.all_stats)))
        random.shuffle(indices)
        for idx in indices[:examples_to_show]:
            self.show_example(self.all_stats[idx])
            print()
        metrics = evaluation.compute_metrics(self.all_stats)
        print("METRICS: %s" % metrics)


def evaluate(args):
    print("Evaluation:")
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    saver.restore_args(args)
    arguments.backport_default_args(args)
    if args.eval_train:
        eval_dataset, _ = dataset.get_dataset(args)
    else:
        eval_dataset = dataset.get_eval_dataset(args)
    m = models.get_model(args)
    if m.last_step == 0:
        raise ValueError('Attempting to evaluate on untrained model')
    m.model.eval()
    executor_cls = executor.get_executor(args)
    if executor_cls:
        current_executor = executor_cls()
    else:
        current_executor = None
    if args.example_id is not None:
        eval_dataset.data = [eval_dataset.task[args.example_id]]

    all_stats = []
    with tqdm.tqdm(total=len(eval_dataset.data)) as pbar:
        m.worker_pool = mp.Pool(mp.cpu_count())
        for stats in evaluation.run_inference(eval_dataset, m, current_executor):
            all_stats.append(stats)
            pbar.update(1)

    report = EvalReport(args.tag, all_stats)
    report.save()
    report.display(examples_to_show=10)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.model_type or (not args.model_dir and args.model_type != 'search'):
        raise ValueError("Specify model_dir and model_type")
    if not args.tag:
        args.tag = args.model_type
    evaluate(args)
