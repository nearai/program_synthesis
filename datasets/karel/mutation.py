import collections
import copy
import itertools

import numpy as np

import parser_for_synthesis

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


conds = [{
    'type': t
}
         for t in ('frontIsClear', 'leftIsClear', 'rightIsClear',
                   'markersPresent', 'noMarkersPresent')]
# no not for markersPresent and noMarkersPresent
conds.extend({'type': 'not', 'cond': cond} for cond in conds[:3])
conds_masked_probs = {
    n: masked_uniform(len(conds), i)
    for i, n in enumerate(
        ('frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent',
         'noMarkersPresent', 'notfrontIsClear', 'notleftIsClear',
         'notrightIsClear'))
}

action_names = ('move', 'turnLeft', 'turnRight', 'putMarker', 'pickMarker')
actions_masked_probs = {
    n: masked_uniform(len(action_names), i)
    for i, n in enumerate(action_names)
}
actions = [{
    'type': t
} for t in ('move', 'turnLeft', 'turnRight', 'putMarker', 'pickMarker')]

repeat_counts = [{'type': 'count', 'value': i} for i in range(2, 11)]
repeat_masked_probs = [None, None] + [masked_uniform(len(repeat_counts), i) for
        i in range(len(repeat_counts))]


def random_singular_block():
    type_ = np.random.choice(('if', 'while', 'repeat'))
    if type_ == 'repeat':
        return {'type': type_, 'times': np.random.choice(repeat_counts)}
    else:
        return {'type': type_, 'cond': np.random.choice(conds)}


ADD_ACTION = 0
REMOVE_ACTION = 1
REPLACE_ACTION = 2
UNWRAP_BLOCK = 3
WRAP_BLOCK = 4
WRAP_IFELSE = 5
REPLACE_COND = 6
SWITCH_IF_WHILE = 7
DEFAULT_PROBS = np.array([.5, 1, 1, 1, .25, .75, 1, 1], dtype=float)

def mutate(tree, probs=None):
    # operations:
    # - Add action
    # - Remove action
    # - Replace action
    # - Unwrap if/ifElse/while/repeat
    # - Wrap with if/while/repeat
    # - Wrap with ifElse
    # - Change condition in if/ifElse/while
    # - Switch between if/while

    if probs is None:
        probs = DEFAULT_PROBS.copy()
    assert len(probs) == 8
    assert tree['type'] == 'run'

    action_locs = []
    cond_locs = []
    all_bodies = []
    unwrappables = []
    all_if_whiles = []

    queue = collections.deque([(tree, (None, None))])
    while queue:
        node, address = queue.popleft()
        if node['type'] == 'ifElse':
            bodies = [node['ifBody'], node['elseBody']]
            unwrappables.append(address)
        elif 'body' in node:
            bodies = [node['body']]
            if address[0]:
                unwrappables.append(address)
        else:
            bodies = []
            action_locs.append(address)
        if 'cond' in node or 'times' in node:
            cond_locs.append(node)
        if node['type'] in ('if', 'while'):
            all_if_whiles.append(node)

        for body in bodies:
            for i, child in enumerate(body):
                queue.append((child, (body, i)))
        all_bodies.extend(bodies)

    bodies = None
    add_locs = [(body, i) for body in all_bodies for i in range(len(body) + 1)]
    remove_locs = [x for x in action_locs if len(x[0]) > 1]

    # wrap_block_choices: (n + 1) choose 2 for each len(body)
    # wrap_ifelse_choices: (n + 1) choose 3 for each len(body)
    wrap_block_choices = np.array([len(body) for body in all_bodies],
            dtype=float)
    wrap_ifelse_choices = wrap_block_choices.copy()
    wrap_block_choices *= (wrap_block_choices + 1)
    wrap_block_choices /= 2
    wrap_ifelse_choices *= (wrap_ifelse_choices + 1) * (
        wrap_ifelse_choices - 1)
    wrap_ifelse_choices /= 6

    probs[ADD_ACTION] *= len(add_locs)
    probs[REMOVE_ACTION] *= len(remove_locs)
    probs[REPLACE_ACTION] *= len(action_locs)
    probs[UNWRAP_BLOCK] *= len(unwrappables)
    probs[WRAP_BLOCK] *= sum(wrap_block_choices)
    probs[WRAP_IFELSE] *= sum(wrap_ifelse_choices)
    probs[REPLACE_COND] *= len(cond_locs)
    probs[SWITCH_IF_WHILE] *=  len(all_if_whiles)
    probs_sum = np.sum(probs)
    if probs_sum == 0:
        raise Exception('No mutation possible')
    probs /= probs_sum

    choice = np.random.choice(8, p=probs)
    if choice == ADD_ACTION:
        body, i = add_locs[np.random.choice(len(add_locs))]
        body.insert(i, np.random.choice(actions))
    elif choice == REMOVE_ACTION:
        body, i = remove_locs[np.random.choice(len(remove_locs))]
        del body[i]
    elif choice == REPLACE_ACTION:
        body, i = action_locs[np.random.choice(len(action_locs))]
        body[i] = np.random.choice(actions,
                p=actions_masked_probs[body[i]['type']])
    elif choice == UNWRAP_BLOCK:
        body, i = unwrappables[np.random.choice(len(unwrappables))]
        block = body[i]
        del body[i]
        body[i:i] = block.get('body', [])
        body[i:i] = block.get('elseBody', [])
        body[i:i] = block.get('ifBody', [])
    elif choice == WRAP_BLOCK:
        wrap_block_choices /= np.sum(wrap_block_choices)
        body = all_bodies[np.random.choice(
            len(all_bodies), p=wrap_block_choices)]
        bounds = list(itertools.combinations(xrange(len(body) + 1), 2))
        left, right = bounds[np.random.choice(len(bounds))]
        subseq = body[left:right]
        del body[left:right]
        new_block = random_singular_block()
        new_block['body'] = subseq
        body.insert(left, new_block)
    elif choice == WRAP_IFELSE:
        wrap_ifelse_choices /= np.sum(wrap_ifelse_choices)
        body = all_bodies[np.random.choice(
            len(all_bodies), p=wrap_ifelse_choices)]
        bounds = list(itertools.combinations(xrange(len(body) + 1), 3))
        left, mid, right = bounds[np.random.choice(len(bounds))]
        if_body = body[left:mid]
        else_body = body[mid:right]
        del body[left:right]
        new_block = {
            'type': 'ifElse',
            'cond': np.random.choice(conds),
            'ifBody': if_body,
            'elseBody': else_body
        }
        body.insert(left, new_block)
    elif choice == REPLACE_COND:
        node = cond_locs[np.random.choice(len(cond_locs))]
        if 'cond' in node:
            node['cond'] = np.random.choice(
                conds,
                p=conds_masked_probs[node['cond']['type'] + node['cond'].get(
                    'cond', {}).get('type', '')])
        elif 'repeat' in node:
            node['repeat'] = np.random.choice(
                    repeat_counts,
                    p=repeat_masked_probs[node['repeat']['times']['value']])
    elif choice == SWITCH_IF_WHILE:
        node = all_if_whiles[np.random.choice(len(all_if_whiles))]
        node['type'] = {'if': 'while', 'while': 'if'}[node['type']]

    return tree


def mutate_n(tree, count, probs=None):
    previous_seqs = set([parser_for_synthesis.tree_to_tokens(tree)])
    for i in range(count):
        found = False
        for _ in range(1000):
            tree = copy.deepcopy(tree)
            mutate(tree, probs)
            new_seq = parser_for_synthesis.tree_to_tokens(tree)
            if new_seq not in previous_seqs:
                previous_seqs.add(new_seq)
                found = True
                break
        if not found:
            raise Exception('Rejection sampling failed')
    return tree


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

