"""
Iterates over the NAPS dataset, converts UAST to C++, compiles and executes it validating that C++ passes all tests.
"""
import tqdm
from itertools import chain
import multiprocessing as mp

from program_synthesis.naps.pipelines.read_naps import read_naps_dataset
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.pipe import Pipe
from program_synthesis.naps.uast.uast_to_cpp import cpp_executor


class FilterPartial(Pipe):
    def __init__(self, is_partial_key):
        self.is_partial_key = is_partial_key

    def __iter__(self):
        for d in self.input:
            if d[self.is_partial_key]:
                continue
            yield d
        return


def compile_program_worker(args):
    program_idx, code_tree, tests = args
    total_num_tests = len(tests)
    sucessful_tests, test_compilation_errors, test_runtime_errors = [0]*3
    try:
        sucessful_tests, test_compilation_errors, test_runtime_errors = cpp_executor.compile_run_program_and_tests(
            code_tree, tests,
            # debug_info=True, cleanup=False
        )
    except (cpp_executor.ProgramCompilationError, cpp_executor.ProgramSourceGenerationError) as e:
        return program_idx, False, str(type(e))
    if test_compilation_errors:
        return program_idx, False, "test_compilation_errors"
    if test_runtime_errors:
        return program_idx, False, "test_runtime_errors"
    return program_idx, sucessful_tests == total_num_tests, "no exception"


if __name__ == "__main__":
    trainA, trainB, test = read_naps_dataset()
    trainB = Compose([
        trainB,
        FilterPartial(is_partial_key="is_partial")
    ])
    pool = mp.Pool()
    map_fn = pool.imap_unordered
    # map_fn = map  # For debugging.
    # Compilation success rate 99%.
    failed = dict()
    failed_num = 0
    total_num = 0

    with trainA, trainB, test, tqdm.tqdm(smoothing=0.1) as pbar:
        for program_idx, is_success, e in map_fn(
                compile_program_worker,
                ((d['solution_id'], d['code_tree'], d['tests'])
                 for program_idx, d in enumerate(chain(trainA, trainB, test))
                 )):
            total_num += 1
            if not is_success:
                failed.setdefault(e, [])
                failed[e].append(program_idx)
                failed_num += 1
            if total_num % 50 == 0:
                pbar.write("Success rate %.6f%% (failed: %d/%d)" % (100.0*(total_num-failed_num)/total_num,
                                                                    failed_num, total_num))
                pbar.write("Failed programs %s" % failed)
            pbar.update(1)
