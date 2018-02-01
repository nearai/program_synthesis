import collections
import traceback

import numpy as np
import pylru

from karel import KarelForSynthesisParser, KarelSyntaxError, TimeoutError


ExecutionResult = collections.namedtuple(
    'ExecutionResult', ['result', 'trace'])


class ExecutorSyntaxException(Exception):
    pass


class ExecutorRuntimeException(Exception):
    pass


def evaluate_code(code, arguments, tests, do_execute):
    stats = {'total': len(tests), 'correct': 0, 'exceptions': 0,
             'result-none': 0, 'syntax-error': 0, 'runtime-exception': 0}
    if not code:
        return stats
    for test in tests:
        try:
            execution_result = do_execute(code, arguments, test['input'])
        except ExecutorSyntaxException:
            stats['syntax-error'] += 1
            continue
        except ExecutorRuntimeException:
            stats['runtime-exception'] += 1
            continue
        except Exception as e:
            print("Exception: %s" % e)
            traceback.print_exc()
            #print(code, arguments, test['input'])
            stats['exceptions'] += 1
            continue
        if execution_result.result is None:
            stats['result-none'] += 1
        if execution_result.result == test['output']:
            stats['correct'] += 1
    return stats


KarelTrace = collections.namedtuple('KarelTrace', ['grids', 'events'])
KarelEvent = collections.namedtuple('KarelEvent', ['timestep', 'type', 'success'])


class KarelExecutor(object):

    def __init__(self, action_limit=1000):
        self.parser = KarelForSynthesisParser()
        self.action_limit = action_limit
        self.code_cache = pylru.lrucache(100000)

    def execute(self, code, arguments, inp, record_trace=False, strict=True):
        if not isinstance(code, tuple):
            code = tuple(code)

        field = np.zeros((15, 18, 18), dtype=np.bool)
        field.ravel()[inp] = True

        trace = []
        successes = []
        steps_taken = [0]
        if record_trace:
            def action_callback(action_name, success, metadata):
                trace.events.append(KarelEvent(timestep=len(trace.grids),
                    type=action_name, success=success))
                trace.grids.append(np.where(field.ravel())[0].tolist())
                successes.append(success)
                steps_taken[0] += 1
                if steps_taken[0] > self.action_limit:
                    raise ExecutorRuntimeException
        else:
            def action_callback(action_name, success, metadata):
                successes.append(success)
                steps_taken[0] += 1
                if steps_taken[0] > self.action_limit:
                    raise ExecutorRuntimeException

        def event_callback(block_name, *args):
            steps_taken[0] += 1
            if steps_taken[0] > self.action_limit:
                raise ExecutorRuntimeException

        self.parser.karel.init_from_array(field)
        self.parser.karel.action_callback = action_callback
        self.parser.karel.event_callback = event_callback
        try:
            if code not in self.code_cache:
                compiled = self.parser.parse(code, debug=False)
                self.code_cache[code] = compiled
            else:
                compiled = self.code_cache[code]
            compiled()
        except KarelSyntaxError:
            raise ExecutorSyntaxException
        except TimeoutError:
            raise ExecutorRuntimeException
        if strict and not all(successes):
            raise ExecutorRuntimeException

        return ExecutionResult(np.where(field.ravel())[0].tolist(), trace)


def get_executor(args):
    if args.dataset.startswith('karel'):
        return KarelExecutor
    return None
