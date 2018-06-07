import numpy as np

from program_synthesis.algolisp.tools import bleu
from program_synthesis.algolisp.dataset import executor


def is_same_code(example, res):
    correct = False
    if hasattr(res, 'code_sequence'):
        if res.code_sequence is not None:
            correct = res.code_sequence == example.code_sequence
        elif res.code_tree is not None:
            correct = res.code_tree == example.code_tree
    else:
        correct = res == example.code_sequence
    return correct


def compute_bleu(example, res):
    try:
        if hasattr(res, 'code_sequence'):
            if res.code_sequence is not None:
                score = bleu.compute_bleu([example.code_sequence], [res.code_sequence])
        else:
            score = bleu.compute_bleu([example.code_sequence], [res])
        return np.asscalar(score)
    except ZeroDivisionError:
        return 0.0


def get_stats_from_code(args):
    res, example, executor_ = args
    if len(example.tests) == 0:
        return None
    if executor_ is not None:
        stats = executor.evaluate_code(
            res.code_tree if res.code_tree else res.code_sequence, example.schema.args, example.tests,
            executor_)
        stats['exact-code-match'] = is_same_code(example, res)
        stats['correct-program'] = int(stats['tests-executed'] == stats['tests-passed'])
    else:
        stats = {'tests-executed': 0, 'tests-passed': 0, 'result-none': 0, 'syntax-error': 0,
                 'runtime-exception': 0, 'exceptions': []}
        stats['correct-program'] = stats['exact-code-match'] = is_same_code(example, res)
    stats['bleu'] = compute_bleu(example, res)
    stats['example'] = example.to_dict()
    stats['res'] = res.to_dict() if hasattr(res, 'to_dict') else res
    return stats


def run_inference(dataset, model, executor_):
    """Runs inference of given model on eval set, and executes resulting code.

    Args:
        dataset: Dataset, iterable of CodeExample to evaluate on.
        model: Model that runs the inference.
        executor: executor class from executor.py.
    """
    for batch in dataset:
        results = model.inference(batch)
        for stats in model.worker_pool.imap(get_stats_from_code, zip(results, batch, [executor_]*len(batch))):
            if stats is not None:
                yield stats
    return


def compute_metrics(all_stats):
    tests_num = 0
    programs_num = 0
    bleu_acc = 0.0
    correct_program_acc = 0
    exact_code_match_acc = 0
    syntax_error_acc = 0
    runtime_exception_acc = 0
    other_exception_acc = 0
    for stats in all_stats:
        tests_num += stats['tests-executed']
        programs_num += 1
        bleu_acc += stats['bleu']
        correct_program_acc += stats['correct-program']
        exact_code_match_acc += stats['exact-code-match']
        syntax_error_acc += stats['syntax-error']
        runtime_exception_acc += stats['runtime-exception']
        other_exception_acc += len(stats['exceptions'])
    return {'bleu': bleu_acc/programs_num,
            'accuracy': correct_program_acc/programs_num,
            'exact_match_accuracy': exact_code_match_acc/programs_num,
            'syntax_error_freq': syntax_error_acc/tests_num,
            'runtime_exception_freq': runtime_exception_acc/tests_num,
            'other_exception_freq': other_exception_acc/tests_num,
            'programs_num': programs_num,
            'tests_num': tests_num
            }
