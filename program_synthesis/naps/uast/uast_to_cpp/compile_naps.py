"""
Iterates over the NAPS dataset, converts UAST to C++, compiles and executes it validating that C++ passes all tests.
"""
import tqdm
from itertools import chain
import multiprocessing as mp

from program_synthesis.naps.pipelines.read_naps import read_naps_dataset
from program_synthesis.naps.pipes.compose import Compose
from program_synthesis.naps.pipes.pipe import Pipe
from program_synthesis.naps.uast.uast_to_cpp.cpp_executor import compile_program


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
    program_idx, code_tree = args
    return program_idx, compile_program(code_tree)


if __name__ == "__main__":
    trainA, trainB, test = read_naps_dataset()
    trainB = Compose([
        trainB,
        FilterPartial(is_partial_key="is_partial")
    ])
    pool = mp.Pool()
    map_fn = pool.imap_unordered
    # map_fn = map  # For debugging.
    failed = []
    total_num = 0
    with trainA, trainB, test, tqdm.tqdm(smoothing=0.001) as pbar:
        for program_idx, is_success in pool.imap_unordered(
                compile_program_worker,ß
                ((program_idx, d['code_tree']) for program_idx, d in enumerate(chain(trainA, trainB, test)))):
            total_num += 1
            if not is_success:
                failed.append(program_idx)
            if total_num % 50 == 0:
                pbar.write("Compilation success rate %.6f%% (failed: %d/%d)" % (100.0*(total_num-len(failed))/total_num,
                                                                                len(failed), total_num))
            pbar.update(1)
    print("Failed programs %s" % failed)
