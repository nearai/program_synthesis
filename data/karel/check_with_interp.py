import argparse
import collections
import json
import itertools
import sys

import numpy as np
import tqdm

sys.path.insert(0, '../../datasets')

from karel.parser_for_synthesis import KarelForSynthesisParser, tree_to_tokens


def str_to_arr(s):
    field = np.zeros((15, 18, 18), dtype=np.bool)
    field.ravel()[[int(v.split(':')[0]) for v in s.split(' ')]] = True
    return field


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    f = open(args.input)
    if args.limit:
        f = itertools.islice(f, args.limit)

    failures = collections.defaultdict(int)
    parser = KarelForSynthesisParser(max_func_call=1000)
    for line in tqdm.tqdm(f):
        obj = json.loads(line)
        code = obj['program_tokens']
        #code = ' '.join(obj['program_tokens'])

        for ex in obj['examples']:
            actions = []
            full_state = str_to_arr(ex['inpgrid_tensor'])
            parser.new_game(
                state=full_state,
                action_callback=lambda *args: actions.append(args))
            parsed = parser.parse(code, debug=False)

            parser.call_counter = [0]
            parsed()
            reconstructed_tokens = [unicode(s) for s in
                    tree_to_tokens(parsed.tree)]
            if reconstructed_tokens != code:
                import IPython
                IPython.embed()

            for name, success in actions:
                if not success:
                    failures[name] += 1

            actions_for_comp = [unicode(name) for name, _ in actions]
            if (not np.all(full_state == str_to_arr(ex['outgrid_tensor'])) or
                    actions_for_comp != ex['actions']):
                print zip(*np.where(
                    full_state != str_to_arr(ex['outgrid_tensor'])))
                import IPython
                IPython.embed()
                sys.exit()

    print('Failures: {}'.format(dict(failures)))
