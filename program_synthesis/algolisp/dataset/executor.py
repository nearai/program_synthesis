import copy
import collections
import traceback

from program_synthesis.algolisp.dataset import code_lisp
from program_synthesis.algolisp.dataset import code_trace
from program_synthesis.algolisp.dataset import code_types
from program_synthesis.algolisp.dataset import data


ExecutionResult = collections.namedtuple('ExecutionResult', ['result', 'trace'])


class ExecutorSyntaxException(Exception):
    pass


class ExecutorRuntimeException(Exception):
    pass


class LispExecutor(object):

    def __init__(self):
        self.lisp_units = code_lisp.load_lisp_units()

    def get_search_args(self, arguments):
        local_units = {k: v for k, v in self.lisp_units.items() if v.compute}
        for name, type_ in arguments.items():
            local_units[name] = (code_lisp.Unit(
                name=name, description=name,
                args=[], return_type=code_types.str_to_type(type_), compute=None, computed_return_type=None))
        return local_units

    def execute(self, code, arguments, inputs, record_trace=True):
        arguments = [(arg, code_types.str_to_type(tp))
                     for arg, tp in arguments.items()]
        if data.is_flat_code(code):
            try:
                code, _ = data.unflatten_code(code, 'lisp')
            except:
                return ExecutionResult(None, None)
        try:
            code_lisp.test_lisp_validity(
                self.lisp_units, code, dict(arguments),
                code_types.Type.T)
        except (ValueError, KeyError, TypeError):
            return ExecutionResult(None, None)
        t = code_trace.CodeTrace() if record_trace else None
        func = code_lisp.compile_func(
            self.lisp_units, 'main', code,
            arguments, code_types.Type.T,
            trace=t)
        try:
            result = func(*[inputs[arg] for arg, _ in arguments])
        except Exception:
            raise ExecutorRuntimeException()
        return ExecutionResult(result, t)

    def compare(self, gold, prediction):
        return gold == prediction


def evaluate_code(code, arguments, tests, executor_):
    stats = {'tests-executed': len(tests), 'tests-passed': 0,
             'result-none': 0, 'syntax-error': 0, 'runtime-exception': 0, 'exceptions': []}
    if not code:
        return stats
    for test in tests:
        try:
            execution_result = executor_.execute(code, arguments, test['input'])
        except ExecutorSyntaxException as e:
            stats['syntax-error'] += 1
            stats['exceptions'].append(str(e))
            continue
        except ExecutorRuntimeException as e:
            stats['runtime-exception'] += 1
            stats['exceptions'].append(str(e))
            continue
        except Exception as e:
            #print("Exception: %s" % e)
            #traceback.print_exc()
            #print(code, arguments, test['input'])
            stats['exceptions'].append(str(e))
            continue
        if execution_result.result is None:
            stats['result-none'] += 1
        if executor_.compare(test['output'], execution_result.result):
            stats['tests-passed'] += 1
    return stats


def get_executor(args):
    if args.dataset in ['metagen', 'handcrafted']:
        return LispExecutor
    return None
