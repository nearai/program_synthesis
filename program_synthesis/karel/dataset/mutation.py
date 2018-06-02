import collections
import copy
import itertools
import struct

import numpy as np

from program_synthesis.karel.dataset import executor
from program_synthesis.karel.dataset import parser_for_synthesis

# Tree structure
# - run: body
# - if: cond, body
# - ifElse: cond, ifBody, elseBody
# - while: cond, body
# - repeat: times, body
# - not: cond


def masked_uniform(choices, i):
    prob = np.full(choices, 1. / (choices - 1))
    prob[i] = 0
    return prob


def choose(rng, sequence, p=None):
    # Using rng.choice directly on sequence sometimes leads to undesirable
    # effects, as that entails converting sequence into a numpy.ndarray.
    return sequence[rng.choice(len(sequence), p=p)]


CONDS = [{
    'type': t
}
         for t in ('frontIsClear', 'leftIsClear', 'rightIsClear',
                   'markersPresent', 'noMarkersPresent')]
# no not for markersPresent and noMarkersPresent
CONDS.extend({'type': 'not', 'cond': cond} for cond in CONDS[:3])
CONDS_MASKED_PROBS = {
    n: masked_uniform(len(CONDS), i)
    for i, n in enumerate(
        ('frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent',
         'noMarkersPresent', 'notfrontIsClear', 'notleftIsClear',
         'notrightIsClear'))
}

ACTION_NAMES = ('move', 'turnLeft', 'turnRight', 'putMarker', 'pickMarker')
ACTIONS_MASKED_PROBS = {
    n: masked_uniform(len(ACTION_NAMES), i)
    for i, n in enumerate(ACTION_NAMES)
}
ACTIONS = [{
    'type': t
} for t in ACTION_NAMES]
ACTIONS_DICT = dict(zip(ACTION_NAMES, ACTIONS))

REPEAT_COUNTS = [{'type': 'count', 'value': i} for i in range(2, 11)]
REPEAT_MASKED_PROBS = [None, None] + [masked_uniform(len(REPEAT_COUNTS), i) for
        i in range(len(REPEAT_COUNTS))]


def random_singular_block(rng):
    type_ = rng.choice(('if', 'while', 'repeat'))
    if type_ == 'repeat':
        return {'type': type_, 'times': rng.choice(REPEAT_COUNTS)}
    else:
        return {'type': type_, 'cond': rng.choice(CONDS)}


# operations:
# - Add action
ADD_ACTION = 0
# - Remove action
REMOVE_ACTION = 1
# - Replace action
REPLACE_ACTION = 2
# - Unwrap if/ifElse/while/repeat
UNWRAP_BLOCK = 3
# - Wrap with if/while/repeat
WRAP_BLOCK = 4
# - Wrap with ifElse
WRAP_IFELSE = 5
# - Change condition in if/ifElse/while
REPLACE_COND = 6
# - Switch between if/while
SWITCH_IF_WHILE = 7
DEFAULT_PROBS = np.array([1, 1, 1, 1, .25, .75, 1, 1], dtype=float)


BodyInfo = collections.namedtuple('BodyInfo', ['node', 'type', 'elems'])


class TreeIndex(object):

    def __init__(self, tree):
        self.action_locs = []
        self.cond_locs = []
        self.all_bodies = []
        self.unwrappables = []
        self.all_if_whiles = []

        queue = collections.deque([(tree, (None, None))])
        while queue:
            node, address = queue.popleft()
            if node['type'] == 'ifElse':
                bodies = [BodyInfo(node, 'ifElse-if', node['ifBody']),
                                   BodyInfo(node, 'ifElse-else', node['elseBody'])]
                self.unwrappables.append(address)
            elif 'body' in node:
                bodies = [BodyInfo(node, node['type'], node['body'])]
                if address[0]:
                    self.unwrappables.append(address)
            else:
                bodies = []
                self.action_locs.append(address)
            if 'cond' in node or 'times' in node:
                self.cond_locs.append(node)
            if node['type'] in ('if', 'while'):
                self.all_if_whiles.append(node)

            for body in bodies:
                for i, child in enumerate(body.elems):
                    queue.append((child, (body.elems, i)))
            self.all_bodies.extend(bodies)

        self.add_locs = [(body.elems, i)
                         for body in self.all_bodies for i in range(
                             len(body) + 1)]
        self.remove_locs = [x for x in self.action_locs if len(x[0]) > 1]

    def count_actions(self):
        # wrap_block_choices: (n + 1) choose 2 for each len(body)
        # wrap_ifelse_choices: (n + 1) choose 3 for each len(body)
        wrap_block_choices = np.array(
            [len(body.elems) for body in tree_index.all_bodies], dtype=float)
        wrap_ifelse_choices = wrap_block_choices.copy()
        wrap_block_choices *= (wrap_block_choices + 1)
        wrap_block_choices /= 2
        wrap_ifelse_choices *= (wrap_ifelse_choices + 1) * (
            wrap_ifelse_choices - 1)
        wrap_ifelse_choices /= 6


def mutate(tree, probs=None, rng=None):
    if probs is None:
        probs = DEFAULT_PROBS.copy()
    if rng is None:
        rng = np.random.RandomState()

    assert len(probs) == 8
    assert tree['type'] == 'run'

    tree_index = TreeIndex(tree)

    # wrap_block_choices: (n + 1) choose 2 for each len(body)
    # wrap_ifelse_choices: (n + 1) choose 3 for each len(body)
    wrap_block_choices = np.array(
        [len(body.elems) for body in tree_index.all_bodies], dtype=float)
    wrap_ifelse_choices = wrap_block_choices.copy()
    wrap_block_choices *= (wrap_block_choices + 1)
    wrap_block_choices /= 2
    wrap_ifelse_choices *= (wrap_ifelse_choices + 1) * (
        wrap_ifelse_choices - 1)
    wrap_ifelse_choices /= 6

    probs[ADD_ACTION] *= len(tree_index.add_locs)
    probs[REMOVE_ACTION] *= len(tree_index.remove_locs)
    probs[REPLACE_ACTION] *= len(tree_index.action_locs)
    probs[UNWRAP_BLOCK] *= len(tree_index.unwrappables)
    probs[WRAP_BLOCK] *= sum(wrap_block_choices)
    probs[WRAP_IFELSE] *= sum(wrap_ifelse_choices)
    probs[REPLACE_COND] *= len(tree_index.cond_locs)
    probs[SWITCH_IF_WHILE] *=  len(tree_index.all_if_whiles)
    probs_sum = np.sum(probs)
    if probs_sum == 0:
        raise Exception('No mutation possible')
    probs /= probs_sum

    choice = rng.choice(8, p=probs)
    if choice == ADD_ACTION:
        body, i = choose(rng, tree_index.add_locs)
        body.insert(i, rng.choice(ACTIONS))
    elif choice == REMOVE_ACTION:
        body, i = choose(rng, tree_index.remove_locs)
        del body[i]
    elif choice == REPLACE_ACTION:
        body, i = choose(rng, tree_index.action_locs)
        body[i] = choose(rng, ACTIONS,
                p=ACTIONS_MASKED_PROBS[body[i]['type']])
    elif choice == UNWRAP_BLOCK:
        body, i = choose(rng, tree_index.unwrappables)
        block = body[i]
        del body[i]
        body[i:i] = block.get('body', [])
        body[i:i] = block.get('elseBody', [])
        body[i:i] = block.get('ifBody', [])
    elif choice == WRAP_BLOCK:
        wrap_block_choices /= np.sum(wrap_block_choices)
        body = choose(rng, tree_index.all_bodies, p=wrap_block_choices).elems
        bounds = list(itertools.combinations(xrange(len(body) + 1), 2))
        left, right = choose(rng, bounds)
        subseq = body[left:right]
        del body[left:right]
        new_block = random_singular_block(rng)
        new_block['body'] = subseq
        body.insert(left, new_block)
    elif choice == WRAP_IFELSE:
        wrap_ifelse_choices /= np.sum(wrap_ifelse_choices)
        body = choose(rng, tree_index.all_bodies, p=wrap_ifelse_choices).elems
        bounds = list(itertools.combinations(xrange(len(body) + 1), 3))
        left, mid, right = choose(rng, bounds)
        if_body = body[left:mid]
        else_body = body[mid:right]
        del body[left:right]
        new_block = {
            'type': 'ifElse',
            'cond': rng.choice(CONDS),
            'ifBody': if_body,
            'elseBody': else_body
        }
        body.insert(left, new_block)
    elif choice == REPLACE_COND:
        node = choose(rng, tree_index.cond_locs)
        if 'cond' in node:
            node['cond'] = rng.choice(
                CONDS,
                p=CONDS_MASKED_PROBS[node['cond']['type'] + node['cond'].get(
                    'cond', {}).get('type', '')])
        elif 'repeat' in node:
            node['repeat'] = rng.choice(
                    REPEAT_COUNTS,
                    p=REPEAT_MASKED_PROBS[node['repeat']['times']['value']])
    elif choice == SWITCH_IF_WHILE:
        node = choose(rng, tree_index.all_if_whiles)
        node['type'] = {'if': 'while', 'while': 'if'}[node['type']]

    return tree


def mutate_n(tree, count, probs=None, rng=None, allow_in_place=False):
    if rng is None:
        rng = np.random.RandomState()
    if count  == 1:
        if allow_in_place:
            return mutate(tree, probs, rng)
        return mutate(copy.deepcopy(tree), probs, rng)

    previous_seqs = set([parser_for_synthesis.tree_to_tokens(tree)])
    for i in range(count):
        found = False
        for _ in range(1000):
            tree = copy.deepcopy(tree)
            mutate(tree, probs, rng)
            new_seq = parser_for_synthesis.tree_to_tokens(tree)
            if new_seq not in previous_seqs:
                previous_seqs.add(new_seq)
                found = True
                break
        if not found:
            raise Exception('Rejection sampling failed')
    return tree


class KarelExampleMutator(object):

    def __init__(self, n_dist, rng_fixed, add_trace, probs=None):
        self.n_dist = n_dist / np.sum(n_dist)
        self.rng_fixed = rng_fixed
        self.add_trace = add_trace
        self.probs = probs

        self.rng = np.random.RandomState()
        self.parser = parser_for_synthesis.KarelForSynthesisParser(
                build_tree=True)
        self.executor = executor.KarelExecutor(action_limit=250)

    def __call__(self, karel_example):
        from ..dataset import KarelExample
        assert karel_example.ref_example is None
        tree = self.parser.parse(karel_example.code_sequence)
        if self.rng_fixed:
            self.rng.seed(int(karel_example.guid[:8], base=16))
        n = self.rng.choice(len(self.n_dist), p=self.n_dist) + 1

        new_tree = mutate_n(tree, n, self.probs, self.rng, allow_in_place=True)
        new_code = parser_for_synthesis.tree_to_tokens(new_tree)

        # TODO: Get the real trace
        new_tests = []
        if self.add_trace:
            for ex in karel_example.input_tests:
                result = self.executor.execute(new_code, None, ex['input'],
                        record_trace=True, strict=True)
                new_ex = dict(ex)
                new_ex['trace'] = result.trace
                new_tests.append(new_ex)

        karel_example.ref_example = KarelExample(
                idx=None,
                guid=None,
                code_sequence=new_code,
                input_tests=new_tests,
                tests=karel_example.tests)
        return karel_example


# Obsolete notes
# ==============
# Actions: move, turnLeft, turnRight, putMarker, pickMarker
# Conditions: frontIsClear, leftIsClear, rightIsClear, markersPresent (+ not)
# Atoms:
# - actions (5)
# - if: pick cond (8) and pick action (5) = 40
# - ifElse: pick cond (8) and pick ifBody (5) and elseBody(5) = 200
#   if nots not allowed, then 100
# - while: pick cond (8) and pick action (5) = 40
# - repeat: pick times (9: 2..10) and body (5)
