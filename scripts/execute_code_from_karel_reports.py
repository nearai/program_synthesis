import collections
import cPickle as pickle
import glob
import json
import os
import re
import sys

import ray
import tqdm

from datasets import executor

# key: GUID
# value: dict where
#   key: program tokens
#   value: pairs of example and [True, False]
execution_cache = collections.defaultdict(dict)

bad_token_re = re.compile(r'<S>|</S>|<UNK>|PL@\d+|\|\|\|')


@ray.remote
def compile_results_for_model(filename, output_path):
    the_executor = executor.KarelExecutor(action_limit=1000)
    f = open(filename)
    f.readline()
    # key: guid
    # value: list of (program, correctness) in order
    results_for_model = collections.defaultdict(list)

    for line in f:
        entry = json.loads(line)
        guid = entry['example']['guid']
        codes = entry['code']['info']['candidates']

        for code in codes:
            code = tuple(code)
            if code in execution_cache[guid]:
                # Results are cached, no need to run it again
                cached_examples, results_for_code = execution_cache[guid][code]
                assert cached_examples == entry['example']['examples']
            else:
                if any(bad_token_re.match(token) for token in code):
                    results_for_code = [False] * len(entry['example'][
                        'examples'])
                else:
                    results_for_code = []
                    for example in entry['example']['examples']:
                        try:
                            log = the_executor.execute(code, None,
                                                       example['in'])
                            results_for_code.append(
                                log.result == example['out'])
                        except (executor.ExecutorSyntaxException,
                                executor.ExecutorRuntimeException) as e:
                            results_for_code.append(False)
                execution_cache[guid][code] = (entry['example']['examples'],
                                               results_for_code)

            results_for_model[guid].append((code, results_for_code))

    with open(output_path, 'w') as f:
        pickle.dump(results_for_model, f, pickle.HIGHEST_PROTOCOL)

@ray.remote
def execute(code, inp, out):
    the_executor = executor.KarelExecutor(action_limit=1000)
    try:
        log = the_executor.execute(code, None, inp)
        return log.result == out
    except (executor.ExecutorSyntaxException,
            executor.ExecutorRuntimeException) as e:
        return False

def compile_results_for_model_local(filename, output_path):
    the_executor = executor.KarelExecutor(action_limit=1000)
    f = open(filename)
    f.readline()
    # key: guid
    # value: list of (program, correctness) in order
    results_for_model = collections.defaultdict(list)

    for line in tqdm.tqdm(f):
        entry = json.loads(line)
        guid = entry['example']['guid']
        codes = entry['code']['info']['candidates']

        for code in codes:
            code = tuple(code)
            if any(bad_token_re.match(token) for token in code):
                results_for_code = [False] * len(entry['example'][
                    'examples'])
            else:
                results_for_code = []
                for example in entry['example']['examples']:
                    results_for_code.append(execute.remote(code,
                    example['in'], example['out']))
            results_for_model[guid].append((code, results_for_code))

    new_results_for_model = collections.defaultdict(list)
    for guid, entries  in tqdm.tqdm(results_for_model.items()):
        for code, objids in entries:
            new_results_for_model[guid].append((code, [(objid if
                isinstance(objid, bool) else ray.get(objid)) for objid
                in objids]))

    with open(output_path, 'w') as f:
        pickle.dump(new_results_for_model, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    #ray.init(driver_mode=ray.PYTHON_MODE)
    ray.init()

    waiting_ids = []

    for fn in sorted(
            #glob.glob('logdirs/20180211/karel-lgrl-ref-edit-m12-sgd-cl1-lr0.1-lds100k-ldr0.5/report-test-*.jsonl')):
            glob.glob('../text2code/models/20180115/karel-sgd-cl1-lr1-lds100k-ldr0.5/report-test-*.jsonl')):
        print fn
        model_name = os.path.basename(os.path.dirname(fn))
        suffix = re.search('report-test-(.*).jsonl', fn).group(1)
        step = re.search(r'report-test.*?-(\d+).jsonl', fn).group(1)
        if int(step) < 1000 or 'karel-sgd-lr' in model_name:
            continue
        output_path = os.path.join(
            os.path.dirname(fn), 'exec-results-test-{}.pkl'.format(suffix))
        #compile_results_for_model_local(fn, output_path)
        waiting_ids.append(compile_results_for_model.remote(fn, output_path))

    pbar = tqdm.tqdm(total=len(waiting_ids))
    while waiting_ids:
        ready_ids, waiting_ids = ray.wait(waiting_ids)
        pbar.update(len(ready_ids))

    #for fn in sorted(
    #        glob.glob('logdirs/*/*/report-test-*.jsonl')):
    #    print fn
    #    model_name = os.path.basename(os.path.dirname(fn))
    #    suffix = re.search('report-test-(.*).jsonl', fn).group(1)
    #    step = re.search(r'report-test-.*?(\d+).jsonl', fn).group(1)
    #    #if int(step) < 1000 or 'karel-sgd-lr' in model_name:
    #    #    continue
    #    output_path = os.path.join(
    #        os.path.dirname(fn), 'exec-results-test-{}.pkl'.format(suffix))
    #    waiting_ids.append(compile_results_for_model.remote(fn, output_path))
