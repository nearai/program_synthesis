from program_synthesis.karel.dataset import executor
from program_synthesis.karel.dataset import parser_for_synthesis

branch_types = {'if', 'ifElse', 'while'}
stmt_types = {'move', 'turnLeft', 'turnRight', 'putMarker', 'pickMarker'}

class CoverageMeasurer(object):
    def __init__(self, code):
        self.parser = parser_for_synthesis.KarelForSynthesisParser(
                build_tree=True)
        self.executor = executor.KarelExecutor()
        self.code = code

        tree = self.parser.parse(code)
        # Statement coverage: actions
        self.stmt_coverage = {span: 0 for span in self.parser.action_spans}
        # Branch coverage: if, ifelse, while
        self.branch_coverage = {(span, cond_value): 0
                                for span in self.parser.cond_block_spans
                                for cond_value in (True, False)}

    def add(self, inp):
        out, trace = self.executor.execute(
                self.code, None, inp, record_trace=True)
        if not out:
            return False
            
        for event in trace.events:
            if event.type in branch_types:
                self.branch_coverage[event.span, event.cond_value] += 1
            elif event.type in stmt_types:
                self.stmt_coverage[event.span] += 1

        return True

    def uncovered(self):
        return (tuple(k for k, v in self.stmt_coverage.iteritems() if v == 0),
                tuple(k for k, v in self.branch_coverage.iteritems() if v == 0))
