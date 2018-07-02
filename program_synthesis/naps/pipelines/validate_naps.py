"""
Iterates over the NAPS dataset and validates that solutions pass all tests.
"""
import tqdm
from itertools import chain
import multiprocessing as mp

from program_synthesis.naps.pipelines.read_naps import read_naps_dataset
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.pipe import Pipe
from program_synthesis.naps.uast.uast_test_config import test_passed

from program_synthesis.naps.uast import Executor


class FilterPartial(Pipe):
    def __init__(self, is_partial_key):
        self.is_partial_key = is_partial_key

    def __iter__(self):
        for d in self.input:
            if d[self.is_partial_key]:
                continue
            yield d
        return


def program_passes_all_tests(args):
    code_tree, tests = args
    for test in tests:
        input_, output_ = test['input'], test['output']
        try:
            ex = Executor(code_tree)
            actual = ex.execute_func("__main__", input_, True)
            if not test_passed(actual, output_):
                return False
        except Exception as e:
            return False
    return True


if __name__ == "__main__":
    trainA, trainB, test = read_naps_dataset()
    trainB = Compose([
        trainB,
        FilterPartial(is_partial_key="is_partial")
    ])
    pool = mp.Pool(max(mp.cpu_count() - 1, 1))
    with trainA, trainB, test, pool, tqdm.tqdm() as pbar:
        num_non_passing_programs = 0
        for res in pool.imap_unordered(program_passes_all_tests,
                                       ((d['code_tree'], d['tests']) for d in chain(trainA, trainB, test))):
            num_non_passing_programs += 1 if not res else 0
            pbar.update(1)
    print("Number of programs that do not pass tests: %d" % num_non_passing_programs)
