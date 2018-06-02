import sys
import os
import collections
import glob
import json
import re
import math
import random
import copy

import numpy as np
import pandas as pd

from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models
from program_synthesis.tools import saver

from program_synthesis.karel.dataset import parser_for_synthesis
from program_synthesis.karel.dataset import executor
from program_synthesis.karel.models import karel_trace_model
from program_synthesis.karel.models.modules import karel_common


def create_trace(grids, actions, conds):
    grids = karel_common.compress_trace(
            [np.where(g.ravel())[0].tolist() for g in
                grids])
    assert len(grids) >= 2
    return executor.KarelTrace(
        grids=grids,
        events=[
            executor.KarelEvent(
                timestep=t,
                type=name,
                span=None,
                cond_span=None,
                cond_value=None,
                success=None)
            for t, name in enumerate(actions)
        ],
        cond_values=conds)


def get_passing_traces(kr, tracer, input_grid, output_grid, candidates):
    some_passed = False
    for cand in candidates:
        tracer.reset(grid=input_grid)
        success = True
        for action in cand:
            success = getattr(kr, action, lambda: False)()
        if  np.all(
                tracer.full_grid == output_grid):
            some_passed = True
            yield create_trace(tracer.grids, tracer.actions, tracer.conds)

    if not some_passed:
        grids = [input_grid, output_grid]
        actions = ['UNK']
        # TODO fix conds
        conds = [tracer.conds[0], tracer.conds[-1]]
        yield create_trace(grids, actions, conds)


def process_traces_infer_results(kr, tracer, batch, inference_results, max_options):
    grid_idx = 0

    input_grids = batch.input_grids.data.numpy().astype(bool)
    output_grids = batch.output_grids.data.numpy().astype(bool)

    output = []
    options_output = []
    for orig_example in batch.orig_examples:
        orig_example = copy.deepcopy(orig_example)
        output.append(orig_example)
        options = [[] for _ in range(6)]
        for i, test in enumerate(orig_example.input_tests + orig_example.tests):
            result = inference_results[grid_idx]
            candidates = result.info['candidates']

            for trace in get_passing_traces(kr, tracer, input_grids[grid_idx], output_grids[grid_idx], candidates):
                options[i].append(trace)
                if max_options == 1:
                    break
            grid_idx += 1
        options_output.append(options)

    for j, example in enumerate(output):
        for k, test in enumerate(example.input_tests + example.tests):
            test['trace'] = options_output[j][k][0]
    yield output
    for i in range(max_options - 1):
        for j, example in enumerate(output):
            for k, test in enumerate(example.input_tests + example.tests):
                test['trace'] = random.choice(options_output[j][k])
        yield output


def score_examples(
        trace_pred_model, code_trace_model, ds, code_trace_executor,
        trace_pred_beam, code_trace_beam, max_options, batch_size=16):
    batch_processor = karel_trace_model.TracePredictionBatchProcessor(trace_pred_model.args, for_eval=True)
    code_trace_batch_processor = karel_trace_model.CodeFromTracesBatchProcessor(code_trace_model.vocab, for_eval=True)
    trace_pred_model.args.max_beam_trees = trace_pred_beam
    code_trace_model.args.max_beam_trees = code_trace_beam
    sum_correct, total = 0, 0
    report = []
    idx = 0
    while idx < len(ds):
        batch_examples = []
        for j in range(batch_size):
            batch_examples.append(ds[idx])
            idx += 1
            if idx >= len(ds):
                break

        batch = batch_processor(batch_examples)
        res = trace_pred_model.inference(batch)
        # new_batch_examples = trace_pred_model.process_infer_results(batch, res)
        # batch = code_trace_batch_processor([dataset.dataset.KarelExample.from_dict(e) for e in new_batch_examples])
        any_correct = [False] * len(batch_examples)
        for new_batch_examples in process_traces_infer_results(
                trace_pred_model.kr, trace_pred_model.tracer, batch, res,
                max_options):
            batch = code_trace_batch_processor(new_batch_examples)                 
            res2 = code_trace_model.inference(batch, filtered=False)
            for j, seq in enumerate(res2):
                ex = batch_examples[j]
                is_correct = check_correct(seq[0], ex.input_tests + ex.tests, code_trace_executor.execute)
                any_correct[j] = any_correct[j] or is_correct
        for is_correct in any_correct:
            if is_correct:
                sum_correct += 1
            report.append((new_batch_examples[j], seq, is_correct))
        total += len(batch_examples)
        print(sum_correct, total, sum_correct / total)
    return report


def check_correct(code, tests, e):
    res = executor.evaluate_code(code, None, tests, e)
    return res['correct'] == len(tests)


def load_model(model_dir, model_type, step=None):
    args = saver.ArgsDict(model_dir=model_dir, model_type=model_type, step=step)
    saver.restore_args(args)
    arguments.backport_default_args(args)
    dataset.set_vocab(args)
    m = models.get_model(args)
    eval_dataset = dataset.get_eval_dataset(args, m)
    m.model.eval()
    the_executor = executor.get_executor(args)()
    return m, eval_dataset, the_executor


def run_eval(trace_pred_path, code_trace_path):
    random.seed(42)
    trace_pred_model, eval_dataset, _ = load_model(trace_pred_path, 'karel-trace-pred', 250100)
    code_trace_model, _, code_trace_executor = load_model(code_trace_path, 'karel-code-trace', 300100)
    report = score_examples(
        trace_pred_model, code_trace_model,
        eval_dataset.dataset, code_trace_executor, 5, 64, 5)
    with open('report.json', 'w') as f:
        f.write(json.dumps(report))


if __name__ == "__main__":
    trace_pred_path = '/home/ubuntu/karel-io-trace-code/logdirs/20180321/karel-trace-pred-gridpresnet'
    code_trace_path = '/home/ubuntu/karel-io-trace-code/logdirs/20180321/karel-code-trace-ioshuf-interleave-nogrid'
    run_eval(trace_pred_path, code_trace_path)

