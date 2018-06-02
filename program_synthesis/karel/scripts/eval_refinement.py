import collections
import cPickle as pickle
import glob
import itertools
import json
import operator
import os
import re
import sys

from program_synthesis.karel.dataset import dataset
from program_synthesis.karel.dataset import executor
from program_synthesis.karel.dataset.karel_runtime import KarelRuntime
from program_synthesis.karel.models import karel_model
from program_synthesis.tools import restore_args

BASE_DIR = ""

with open(BASE_DIR + "text2code-models/karel-sgd-cl1-lr1-lds100k-ldr0.5/report-dev-00100100.jsonl") as f:
    baseline_report = []
    print(f.readline())
    for line in f:
        baseline_report.append(json.loads(line))

class Args(object):
    model_dir = BASE_DIR + 'program_synthesis-models/karel-lgrl-ref-m123-sgd-cl1-lr0.1-lds100k-ldr0.5'
    step = 250100

args = Args()
restore_args(args)
args.word_vocab = ',,/data/karel/word.vocab'
m = karel_model.KarelLGRLRefineModel(args)

batch_processor = m.batch_processor(for_eval=True)

m.args.max_beam_trees = 64
m.args.max_eval_trials = 64

i = 0
result = []
while i < len(baseline_report):
    batch = []
    while len(batch) < 32 and i < len(baseline_report):
        if baseline_report[i]['code']['info']['trees_checked'] == 1:
            i += 1
            continue
        e = dataset.KarelExample.from_dict(baseline_report[i]['example'])
        ref_code_sequence = baseline_report[i]['code']['info']['candidates'][0]
        e.ref_example = dataset.KarelExample(idx=None, guid=None, code_sequence=ref_code_sequence, input_tests=e.input_tests, tests=e.tests)
        batch.append(e)
        i += 1
    print("Starting batch (%d)..." % i)
    res = m.inference(batch_processor(batch))
    for example, infer in zip(batch, res):
        result.append((example, infer))
#    if i > 100:
#        break
print(len(result), len(baseline_report))

the_executor = executor.KarelExecutor()
stats = {'total': len(result), 'fixed': 0}
refinement_results = []
for example, infer in result:
    if not infer.code_sequence:
        continue
    correct = True
    for test in example.input_tests + example.tests:
        try:
            log = the_executor.execute(infer.code_sequence, None, test['input'])
            if log.result != test['output']:
                correct = False
                break
        except (executor.ExecutorRuntimeException, executor.ExecutorSyntaxException) as e:
            correct = False
            break
    refinement_results.append(correct)
    if correct:
        stats['fixed'] += 1

print(float(stats['fixed']) / stats['total'], stats['fixed'], stats['total'])

