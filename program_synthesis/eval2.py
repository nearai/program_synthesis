import argparse
import json
import sys

import numpy as np
import torch
import tqdm

from program_synthesis import arguments
from program_synthesis import datasets
from program_synthesis import models
from program_synthesis import tools
from program_synthesis.datasets import executor
from program_synthesis.tools import evaluation


class EmptyArgs(object):
    pass


def evaluate(args):
    print("Evaluation:")
    print("\tModel type: %s\n\tModel path: %s" %
          (args.model_type, args.model_dir))
    tools.restore_args(args)
    arguments.backport_default_args(args)
    datasets.set_vocab(args)
    m = models.get_model(args)
    if args.eval_final:
        eval_dataset = datasets.get_eval_final_dataset(args, m)
    elif args.eval_train:
        eval_dataset = datasets.get_train_dataset(args, m, for_eval=True)
    else:
        eval_dataset = datasets.get_eval_dataset(args, m)
    if m.last_step == 0:
        raise ValueError('Attempting to evaluate on untrained model')
    m.model.eval()
    the_executor = executor.get_executor(args)()

    top_k_exact = np.zeros(args.max_beam_trees, dtype=int)
    top_k_sem = np.zeros(args.max_beam_trees, dtype=int)
    top_k_gen = np.zeros(args.max_beam_trees, dtype=int)
    total = 0.0

    iterator = tqdm.tqdm(eval_dataset)
    for batch in iterator:
        sequences = m.inference(batch, filtered=False)

        for batch_idx, beams in enumerate(sequences):
            total += 1
            orig_example = batch.orig_examples[batch_idx]
            exact_found, sem_found, gen_found = False, False, False
            for rank, tokens in enumerate(beams):
                # Exact match
                if not exact_found and tokens == orig_example.code_sequence:
                    top_k_exact[rank:] += 1
                    exact_found = True

                if not sem_found or not exact_found:
                    # Semantic match (passes all input tests)
                    input_tests_eval = executor.evaluate_code(
                        tokens, None, orig_example.input_tests,
                        the_executor.execute)
                    sem_match = input_tests_eval['correct'] == len(
                            orig_example.input_tests)
                    if not sem_found and sem_match:
                        top_k_sem[rank:] += 1
                        sem_found = True

                    # Generalization (passes all input tests + other tests)
                    tests_eval = executor.evaluate_code(
                        tokens, None, orig_example.tests, the_executor.execute)
                    gen = sem_match and tests_eval['correct'] == len(
                            orig_example.tests)
                    if not gen_found and gen:
                        top_k_gen[rank:] += 1
                        gen_found = True

                if exact_found and sem_found and gen_found:
                    break

        iterator.set_postfix(
            exact=top_k_exact[0] / total,
            sem=top_k_sem[0] / total,
            gen=top_k_gen[0] / total)

    with open(args.report_path, 'w') as f:
        json.dump({
            'total': total,
            'exact': top_k_exact.tolist(),
            'semantic': top_k_sem.tolist(),
            'generalization': top_k_gen.tolist()
        }, f)



if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')
    args = parser.parse_args()

    assert args.report_path

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.model_type or (not args.model_dir and
                               args.model_type != 'search'):
        raise ValueError("Specify model_dir and model_type")
    if not args.tag:
        args.tag = args.model_type
    evaluate(args)
