import sys
import os
import collections
import glob
import json
import re
import tqdm

import numpy as np
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

def eval_traces(trace_model, eval_dataset, beam_size):
    parser = parser_for_synthesis.KarelForSynthesisParser()
    tracer = karel_trace_model.KarelTracer(parser.karel)
    def get_trace(tracer, prog, input_grid):
        tracer.reset(grid=input_grid)
        prog()
        return tracer.actions
        
    def check_trace_passes(kr, tracer, input_grid, output_grid, candidates, gold_trace):
        passes = collections.defaultdict(bool)
        exact_match = collections.defaultdict(bool)
        for i, cand in enumerate(candidates):
            # print(cand, gold_trace)
            if cand == gold_trace:
                exact_match[i] = True
            tracer.reset(grid=input_grid)
            for action in cand:
                success = getattr(kr, action, lambda: False)()
            if np.all(tracer.full_grid == output_grid):
                passes[i] = True
                break
        return passes, exact_match

    trace_model.args.max_beam_trees = beam_size
    report = []
    iterator = tqdm.tqdm(eval_dataset, dynamic_ncols=True)
    for batch in iterator:
        res = trace_model.inference(batch)
        idx = 0
        input_grids = batch.input_grids.data.numpy().astype(bool)
        output_grids = batch.output_grids.data.numpy().astype(bool)
        for ex in batch.orig_examples:
            info = []
            prog = parser.parse(ex.code_sequence)
            for i, test in enumerate(ex.input_tests + ex.tests):
                candidates = res[idx].info['candidates']
                gold_trace = get_trace(tracer, prog, input_grids[idx])
                passes, exact_match = check_trace_passes(trace_model.kr, trace_model.tracer, input_grids[idx], output_grids[idx], candidates, gold_trace)
                info.append((passes, exact_match))
                idx += 1
            report.append(info)
        assert idx == len(res)
    return report

def at_least_k(dct, k):
    for i in range(k):
        if dct[i]:
            return True
    return False

def show_report(report):
    for k in [1, 5, 10, 20, 30, 32, 40, 50, 60, 64]:
        correct = 0.0
        exact_match = 0.0
        for row in report:
            if all([at_least_k(x[0], k) for x in row]):
                correct += 1
            if all([at_least_k(x[1], k) for x in row]):
                exact_match += 1
        print('Top %d' % k, correct, len(report), correct / len(report), exact_match, exact_match / len(report))
    

if __name__ == "__main__":
    trace_model, trace_eval_dataset, trace_executor = load_model(
        '../../../../karel-io-trace-code/logdirs/20180321/karel-trace-pred-gridpresnet',
        'karel-trace-pred', 300100)
    report = eval_traces(trace_model, trace_eval_dataset, 20)
    show_report(report)

