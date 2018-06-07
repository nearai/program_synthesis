import argparse
import functools
import sys
import os
import json

import torch

from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models
from program_synthesis.karel.dataset import executor
from program_synthesis import tools
from program_synthesis.algolisp.tools import evaluation


def evaluate(args):
    print("Evaluation:")
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    tools.restore_args(args)
    arguments.backport_default_args(args)
    dataset.set_vocab(args)
    m = models.get_model(args)
    if args.eval_final:
        eval_dataset = dataset.get_eval_final_dataset(args, m)
    elif args.eval_train:
        eval_dataset = dataset.get_train_dataset(args, m, for_eval=True)
    else:
        eval_dataset = dataset.get_eval_dataset(args, m)
    if m.last_step == 0:
        raise ValueError('Attempting to evaluate on untrained model')
    m.model.eval()
    current_executor = executor.get_executor(args)()
    if args.example_id is not None:
        eval_dataset.data = [eval_dataset.task[args.example_id]]

    evaluation.run_eval(
        args.tag, eval_dataset, m.inference,
        current_executor.execute, not args.hide_example_info,
        args.report_path)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.model_type or (not args.model_dir and args.model_type != 'search'):
        raise ValueError("Specify model_dir and model_type")
    if not args.tag:
        args.tag = args.model_type
    evaluate(args)
