import argparse
import json
import sys

import numpy as np
import torch
import tqdm

from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models
from program_synthesis.karel.dataset import executor
from program_synthesis import tools
from program_synthesis.algolisp.tools import evaluation

class EmptyArgs(object):
    pass


def evaluate(args):
    print("Evaluation:")
    print("\tModel type: %s\n\tModel path: %s" %
          (args.model_type, args.model_dir))
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
    the_executor = executor.get_executor(args)()

    top_k_exact = np.zeros(args.max_beam_trees, dtype=int)
    top_k_sem = np.zeros(args.max_beam_trees, dtype=int)
    top_k_gen = np.zeros(args.max_beam_trees, dtype=int)
    exact_ranks = []
    sem_ranks = []
    gen_ranks = []
    outputs = []
    total = 0.0

    iterator = tqdm.tqdm(eval_dataset, dynamic_ncols=True)
    for batch in iterator:
        sequences = m.inference(batch, filtered=False)

        for batch_idx, beams in enumerate(sequences):
            total += 1
            orig_example = batch.orig_examples[batch_idx]
            exact_found, sem_found, gen_found = False, False, False
            exact_rank, sem_rank, gen_rank = None, None, None
            for rank, tokens in enumerate(beams):
                # Exact match
                ref_code = getattr(orig_example, 'code_sequence',
                                   getattr(orig_example, 'goal_code', None))
                if not exact_found and tuple(tokens) == tuple(ref_code):
                    top_k_exact[rank:] += 1
                    exact_found = True
                    exact_rank = rank

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
                        sem_rank = rank

                    # Generalization (passes all input tests + other tests)
                    tests_eval = executor.evaluate_code(
                        tokens, None, orig_example.tests, the_executor.execute)
                    gen = sem_match and tests_eval['correct'] == len(
                            orig_example.tests)
                    if not gen_found and gen:
                        top_k_gen[rank:] += 1
                        gen_found = True
                        gen_rank = rank

                if exact_found and sem_found and gen_found:
                    break

            exact_ranks.append(exact_rank)
            sem_ranks.append(sem_rank)
            gen_ranks.append(gen_rank)
            if args.save_beam_outputs:
                outputs.append(beams)

        iterator.set_postfix(
            exact=top_k_exact[0] / total,
            sem=top_k_sem[0] / total,
            gen=top_k_gen[0] / total)

    with open(args.report_path, 'w') as f:
        json.dump({
            # Total number of programs in this report.
            'total': total,
            # List where the Nth entry contains the number of programs with
            # exact match among the top N beam search outputs.
            # Length = args.max_beam_trees.
            'exact': top_k_exact.tolist(),
            # List where the Nth entry contains the number of programs with
            # semantic match (correct on all `input_tests`, given to the model)
            # among the top N beam search outputs.
            # Length = args.max_beam_trees.
            'semantic': top_k_sem.tolist(),
            # List where the Nth entry contains the number of programs with
            # generalization (correct on all `input_tests` and `tests) among the
            # top N beam search outputs.
            # Length = args.max_beam_trees.
            'generalization': top_k_gen.tolist(),
            # For each program, the rank at which the corresponding type of
            # match was found (None if not applicable to any rank).
            'ranks': {
                'exact': exact_ranks,
                'semantic': sem_ranks,
                'generalization': gen_ranks,
            },
            # List of length `total` where each item is a list containing
            # `args.max_beam_trees` programs, as output by the beam search.
            # Can be empty if this output was not saved.
            'beam_outputs': outputs
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
