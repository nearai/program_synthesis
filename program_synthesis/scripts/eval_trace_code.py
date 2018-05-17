import sys
import os
import collections
import glob
import json
import re
import math

import pandas as pd

from program_synthesis import arguments
from program_synthesis import datasets
from program_synthesis import models
from program_synthesis.tools import saver

from program_synthesis.datasets.karel import parser_for_synthesis
from program_synthesis.datasets import executor
from program_synthesis.models import karel_trace_model


def load_model(model_dir, model_type, step=None):
    args = saver.ArgsDict(model_dir=model_dir, model_type=model_type, step=step)
    saver.restore_args(args)
    arguments.backport_default_args(args)
    datasets.set_vocab(args)
    m = models.get_model(args)
    eval_dataset = datasets.get_eval_dataset(args, m)
    m.model.eval()
    the_executor = executor.get_executor(args)()
    return m, eval_dataset, the_executor


parser = parser_for_synthesis.KarelForSynthesisParser()
tracer = karel_trace_model.KarelTracer(parser.karel)

def get_traces(code, tests):
    result = []
    prog = parser.parse(code)
    for test in tests:
        tracer.reset(indices=test['input'])
        prog()
        result.append(tracer.actions)
    return result


def check_correct(code, tests, e):
    res = executor.evaluate_code(code, None, tests, e)
    return res['correct'] == len(tests)


def load_report(report, ds, executor, traces=True):
  result = []
  idx = 0
  total_correct = 0
  for ex in ds.dataset:
    seq = report['beam_outputs'][idx][0]
    match = seq == ex.code_sequence
    if traces:
      gt_traces = get_traces(ex.code_sequence, ex.input_tests)
      inferred_traces = []
      # inferred_traces = [[ev.type for ev in test['trace'].events] for test in ex.input_tests]
    else:
      gt_traces, inferred_traces = [], []
    input_correct = check_correct(seq, ex.input_tests, executor.execute)
    all_correct = check_correct(seq, ex.input_tests + ex.tests, executor.execute)
    result.append((' '.join(ex.code_sequence), ' '.join(seq), gt_traces, inferred_traces, match, input_correct, all_correct))
    total_correct += 1 if all_correct else 0
    idx += 1
  print(float(total_correct) / len(result))
  return result


def show_stratified_report(reports):
  def length_to_key(l):
    if l < 15:
        return '0-15'
    elif l < 30:
        return '15-30'
    else:
        return '30+'
  result = []
  for rows in reports:
    total_by_length, correct_by_length = collections.defaultdict(int), collections.defaultdict(int)
    for r in rows:
        l = len(r[0].split(' ') if isinstance(r[0], str) else r[0])
        key = length_to_key(l)
        total_by_length[key] += 1
        if r[-1]:
            correct_by_length[key] += 1

    q = []
    for l in sorted(total_by_length.keys()):
      q.append((
            l, total_by_length[l] / len(rows), 
            float(correct_by_length[l]) / total_by_length[l], 
            correct_by_length[l], total_by_length[l]))
    def contain_and_not(seq, contain, not_contain):
        has_contain = False if contain else True
        has_not_contain = False
        for x in contain:
            if x in seq:
                has_contain = True
                break
        for x in not_contain:
            if x in seq:
                has_not_contain = True
                break
        return has_contain and not has_not_contain

    stats = collections.defaultdict(lambda: [0, 0])
    keys = [
        ('COND', ['IF', 'IFELSE'], ['REPEAT', 'WHILE']),
        ('LOOP', ['REPEAT', 'WHILE'], ['IF', 'IFELSE']),
        ('NONE', [], ['IF', 'IFELSE', 'REPEAT', 'WHILE']),
        ('COND_LOOP', ['IF', 'IFELSE', 'REPEAT', 'WHILE'], [])
    ]
    for r in rows:
        for key, contain, not_contain in keys:
            if contain_and_not(r[0], contain, not_contain):
                stats[key][0] += 1
                if r[-1]:
                    stats[key][1] += 1
        key = 'LOOP-%d' % (r[0].count('REPEAT') + r[0].count('WHILE'))
        stats[key][0] += 1
        if r[-1]:
            stats[key][1] += 1

    for key in sorted(list(stats.keys())):
        q.append((
            key, stats[key][0] / len(rows), 
            float(stats[key][1]) / stats[key][0], stats[key][1],
            stats[key][0]))
    result.append(q)
  return result


if __name__ == "__main__":
    m, eval_dataset, the_executor = load_model(
        '/home/ubuntu/karel-io-trace-code/logdirs/20180321/karel-code-trace-ioshuf-interleave-nogrid',
        'karel-code-trace', step=300100)
    orig_report = json.load(open('/home/ubuntu/karel-io-trace-code/logdirs/20180321/karel-code-trace-ioshuf-interleave-nogrid/report-20180321-val-bs43-300100.jsonl'))
    baseline_report = json.load(open('/home/ubuntu/karel-io-trace-code/baseline-msr.eval2.json'))
    iotracecode_result = load_report(orig_report, eval_dataset, the_executor, True)
    baseline_result = load_report(baseline_report, eval_dataset, the_executor)
    report = show_stratified_report([baseline_result, iotracecode_result])
    for rows in report:
        for row in rows:
            print('\t'.join([str(x) for x in row]))

