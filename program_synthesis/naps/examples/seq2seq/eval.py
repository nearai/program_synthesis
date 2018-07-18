import time
import random
import json
import os

import torch
import tqdm
import multiprocessing as mp

from program_synthesis.common.tools import saver
from program_synthesis.naps.uast import lisp_to_uast, uast_pprint
from program_synthesis.naps.pipes.basic_pipes import JsonLoader, Batch
from program_synthesis.naps.pipes.compose import Compose

from program_synthesis.naps.examples.seq2seq import executor, evaluation, arguments
from program_synthesis.naps.examples.seq2seq.pointer_seq2seq_model import PointerSeq2SeqModel
from program_synthesis.naps.examples.seq2seq.pipes import SortBatchByLen, SplitTests, SkipPartial


BASE_PATH = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_PATH, "../../../../data/naps/")
VERSION = "1.0"
TEST_PATH = os.path.join(DATA_FOLDER, "naps.test.{version}.jsonl".format(version=VERSION))
WORD_VOCAB_PATH = os.path.join(BASE_PATH, "vocabs/word.vocab")
CODE_VOCAB_PATH = os.path.join(BASE_PATH, "vocabs/code.vocab")


def read_naps_dataset_batched(batch_size=100):
    test = Compose([
        open(TEST_PATH),
        JsonLoader(),
        SkipPartial(is_partial_key="is_partial"),
        SplitTests(tests_key="tests", input_tests_key="search_tests", eval_tests_key="eval_tests"),
        Batch(batch_size=batch_size),
        SortBatchByLen(key="text")
    ])
    return test


class EvalReport(object):

    def __init__(self, tag, all_stats):
        self.tag = tag
        self.all_stats = all_stats
        timestamp = int(time.time())
        self.report_path = 'reports/report-%s-%s.json' % (self.tag, timestamp)

    def save(self):
        with open(self.report_path, 'w') as f:
            metrics = evaluation.compute_metrics(self.all_stats)
            f.write(json.dumps(metrics) + "\n")
            for stats in self.all_stats:
                f.write(json.dumps(stats) + "\n")

    def show_example(self, stats):
        example = stats['example']
        res = stats['res']
        golden_lines, inferred_lines = None, None
        try:
            ex_uast = lisp_to_uast.lisp_to_uast(example['code_sequence'])
            res_uast = lisp_to_uast.lisp_to_uast(res['code_sequence'])
            golden_lines = uast_pprint.pformat(ex_uast)
            inferred_lines = uast_pprint.pformat(res_uast)
        except:
            golden_lines = example['code_sequence']
            inferred_lines = res['code_sequence']
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
    print("\tModel path: %s" % args.model_dir)
    saver.restore_args(args)
    args.word_vocab = WORD_VOCAB_PATH
    args.code_vocab = CODE_VOCAB_PATH
    arguments.backport_default_args(args)
    eval_dataset = read_naps_dataset_batched(args.batch_size)
    m = PointerSeq2SeqModel(args)
    if m.last_step == 0:
        raise ValueError('Attempting to evaluate on untrained model')
    m.model.eval()
    current_executor = executor.UASTExecutor()

    all_stats = []
    total_inferences = 0
    correct_inferences = 0
    with eval_dataset, torch.no_grad(), tqdm.tqdm() as pbar:
        m.worker_pool = mp.Pool(mp.cpu_count())
        for stats in evaluation.run_inference(eval_dataset, m, current_executor):
            all_stats.append(stats)
            pbar.update(1)
            total_inferences += 1
            correct_inferences += stats['correct-program']
            pbar.write('Accuracy:\t%.6f'% (1.0*correct_inferences/total_inferences))

    report = EvalReport('seq2seq', all_stats)
    report.save()
    report.display(examples_to_show=10)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    evaluate(args)
