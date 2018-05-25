

class CodeTrace(object):

    def __init__(self):
        self.clear()

    def _get_callable(self, func):
        if func not in self.names:
            self.names[func] = 'LAMBDA%d' % (len(self.names) + 1)
        return self.names[func]

    def add_call(self, func_call, args):
        if func_call in ('lambda1', 'lambda2'):
            return -1
        args = [self._get_callable(arg) if callable(arg) else arg for arg in args]
        self.history.append((func_call, args))
        self.results.append(None)
        return len(self.history) - 1

    def add_result(self, id, result):
        if id < 0:
            return
        self.results[id] = result

    def clear(self):
        self.history = []
        self.results = []
        self.names = {}
