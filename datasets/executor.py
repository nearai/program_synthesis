import collections

import numpy as np

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

    def __init__(self):
        self.parser = KarelForSynthesisParser(max_func_call=1000)

    def execute(self, code, arguments, inp, record_trace=False):
        field = np.zeros((15, 18, 18), dtype=np.bool)
        field.ravel()[inp] = True

        trace = []
        successes = []
        if record_trace:
            def action_callback(action_name, success):
                trace.events.append(KarelEvent(timestep=len(trace.grids),
                    type=action_name, success=success))
                trace.grids.append(np.where(field.ravel())[0].tolist())
                successes.append(success)
        else:
            def action_callback(action_name, success):
                pass

        # TODO provide code as list instead of as string
        self.parser.new_game(
            state=field,
            action_callback=action_callback)
        try:
            self.parser.run(' '.join(code), debug=False)
        except KarelSyntaxError:
            raise ExecutorSyntaxException
        except TimeoutError:
            raise ExecutorRuntimeException

        return ExecutionResult(np.where(field.ravel())[0].tolist(), trace)


def get_executor(args):
    if args.dataset.startswith('karel'):
        return KarelExecutor
    return None
