import argparse
import collections
import json
import itertools
import sys

import numpy as np
import tqdm

from program_synthesis.karel.dataset.parser_for_synthesis import KarelForSynthesisParser, tree_to_tokens


def str_to_arr(s):
    field = np.zeros((15, 18, 18), dtype=np.bool)
    field.ravel()[[int(v.split(':')[0]) for v in s.split(' ')]] = True
    return field


class Timeout(object):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.steps = 0

    def inc(self):
        self.steps += 1
        if self.steps >= self.max_steps:
            raise TimeoutError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    f = open(args.input)
    if args.limit:
        f = itertools.islice(f, args.limit)

    failures = collections.defaultdict(int)
    parser_tree = KarelForSynthesisParser(build_tree=True)
    parser = KarelForSynthesisParser(build_tree=False)
    for line in tqdm.tqdm(f):
        obj = json.loads(line)
        code = obj['program_tokens']
        tree = parser_tree.parse(code, debug=False)
        reconstructed_tokens = [unicode(s) for s in tree_to_tokens(tree)]
        if reconstructed_tokens != code:
            import IPython
            IPython.embed()
        parsed = parser.parse(code, debug=False)

        for ex in obj['examples']:
            actions = []
            timeout = Timeout(1000)
            full_state = str_to_arr(ex['inpgrid_tensor'])

            def action_callback(name, success, metadata):
                actions.append(unicode(name))
                if not success:
                    failures[name] += 1
                timeout.inc()
            def event_callback(*args):
                timeout.inc()


            parser.karel.init_from_array(full_state)
            parser.karel.action_callback = action_callback
            parser.karel.event_callback = event_callback
            parsed()

            if (not np.all(full_state == str_to_arr(ex['outgrid_tensor'])) or
                    actions != ex['actions']):
                print(zip(*np.where(
                    full_state != str_to_arr(ex['outgrid_tensor']))))
                import IPython
                IPython.embed()
                sys.exit()

    print('Failures: {}'.format(dict(failures)))
