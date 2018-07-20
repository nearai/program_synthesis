import copy
import collections


from program_synthesis.naps import uast
from program_synthesis.naps.uast import lisp_to_uast
from program_synthesis.naps.uast import uast_test_config

ExecutionResult = collections.namedtuple('ExecutionResult', ['result', 'trace'])


class ExecutorSyntaxException(Exception):
    pass


class ExecutorRuntimeException(Exception):
    pass


class UASTExecutor(object):

    def execute(self, code, inputs):
        if not isinstance(code, dict):
            try:
                code = lisp_to_uast.lisp_to_uast(code)
            except lisp_to_uast.LispSyntaxException:
                raise ExecutorSyntaxException()
        e = uast.Executor(code)
        trace_ = None
        try:
            assert isinstance(inputs, list)
            inputs = copy.deepcopy(inputs)
            try:
                result = e.execute_func("__main__", inputs, True)
            except KeyboardInterrupt:
                print(code)
                print(inputs)
                raise
        except Exception as e:
            raise ExecutorRuntimeException(e)
        return ExecutionResult(result, trace_)

    def compare(self, gold, prediction):
        return uast_test_config.test_passed(gold, prediction)


def evaluate_code(code, tests, executor_):
    stats = {'tests-executed': len(tests), 'tests-passed': 0,
             'result-none': 0, 'syntax-error': 0, 'runtime-exception': 0, 'exceptions': []}
    if not code:
        return stats
    for test in tests:
        try:
            execution_result = executor_.execute(code, test['input'])
        except ExecutorSyntaxException as e:
            stats['syntax-error'] += 1
            stats['exceptions'].append(str(e))
            continue
        except ExecutorRuntimeException as e:
            stats['runtime-exception'] += 1
            stats['exceptions'].append(str(e))
            continue
        except Exception as e:
            stats['exceptions'].append(str(e))
            continue
        if execution_result.result is None:
            stats['result-none'] += 1
        if executor_.compare(test['output'], execution_result.result):
            stats['tests-passed'] += 1
    return stats


