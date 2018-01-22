import collections

import numpy as np

from karel import KarelForSynthesisParser, KarelSyntaxError, TimeoutError


ExecutionResult = collections.namedtuple(
    'ExecutionResult', ['result', 'trace'])


class ExecutorSyntaxException(Exception):
    pass


class ExecutorRuntimeException(Exception):
    pass


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
                trace.append((action_name, success, field.copy()))
                successes.append(success)
        else:
            def action_callback(action_name, success):
                successes.append(success)

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
        # TODO Check whether we should raise exception for this.
        if not np.all(successes):
            raise ExecutorRuntimeException

        return ExecutionResult(np.where(field.ravel())[0].tolist(), trace)


def get_executor(args):
    if args.dataset in ['karel']:
        return KarelExecutor
    return None
