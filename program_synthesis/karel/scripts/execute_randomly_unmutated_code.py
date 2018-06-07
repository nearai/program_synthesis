import collections
import cPickle as pickle
import glob
import json
import os
import re
import sys

import tqdm
import numpy as np

from program_synthesis.karel import arguments
from program_synthesis.algolisp.tools import evaluation
from program_synthesis.karel.dataset import executor, dataset
from program_synthesis.karel.dataset import mutation, parser_for_synthesis


class DummyModel(object):
    def batch_processor(self, for_eval):
        return lambda x: x


if __name__ == '__main__':
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')
    parser.add_argument('--mutate-n', type=int, required=True)
    parser.add_argument('--trials', type=int, required=True)
    args = parser.parse_args()
    args.dataset = 'karel'
    args.karel_trace_enc = 'none'
    args.load_sync = True

    ds = dataset.get_karel_eval_final_dataset(args, DummyModel())
    the_executor = executor.KarelExecutor(action_limit=1000)
    parser = parser_for_synthesis.KarelForSynthesisParser(
                    build_tree=True)

    total, correct, match = 0, 0, 0
    rng = np.random.RandomState(112358)
    for batch in tqdm.tqdm(ds):
        for item in batch:
            tried_code = set()
            orig_tree = parser.parse(item.ref_example.code_sequence)
            match_found, correct_found = False, False
            total += 1
            for t in range(args.trials):
                mut_success = False
                for i in range(1000):
                    tree = mutation.mutate_n(
                        orig_tree,
                        args.mutate_n,
                        rng=rng,
                        allow_in_place=False)
                    code = parser_for_synthesis.tree_to_tokens(tree)
                    if code not in tried_code:
                        tried_code.add(code)
                        mut_success = True
                        break
                if not mut_success:
                    break
                #tree = mutation.mutate_n(
                #    orig_tree,
                #    args.mutate_n,
                #    rng=rng,
                #    allow_in_place=False)
                #code = parser_for_synthesis.tree_to_tokens(tree)
                if tuple(str(t) for t in code) == tuple(str(t) for t in
                        item.code_sequence):
                    match_found = True
                results_for_code = []
                for example in (item.input_tests + item.tests):
                    try:
                        log = the_executor.execute(code, None,
                                                   example['input'])
                        results_for_code.append(
                            log.result == example['output'])
                    except (executor.ExecutorSyntaxException,
                            executor.ExecutorRuntimeException) as e:
                        results_for_code.append(False)
                if all(results_for_code):
                    correct_found = True
                if match_found and correct_found:
                    break
            correct += correct_found
            match += match_found

    print('Gen', correct / float(total), correct, total)
    print('Exact', match / float(total), match, total)
