import collections

import gym

from program_synthesis.datasets import executor as executor_mod
from program_synthesis.datasets.karel import mutation
from program_synthesis.datasets.karel import parser_for_synthesis


class MutationActionSpace(gym.Space):
    parser = parser_for_synthesis.KarelForSynthesisParser(build_tree=True)

    def __init__(self, tree=None, code=None):
        assert (tree is None) ^ (code is None)
        if tree:
            self.tree = tree
            self.code = parser_for_synthesis.tree_to_tokens(tree)
        else:
            self.tree = self.parser.parse(code)
            self.code = code

        self.tree_index = tree_index = mutation.TreeIndex(self.tree)

        # points to token before insert point
        #           v    v                                      v
        # DEF run m( move IF c( markersPresent c) i( turnLeft i) m)
        #         0  1                                        2
        self.pre_insert_locs = {}
        # points to token after insert point
        #           v    v                                      v
        # DEF run m( move IF c( markersPresent c) i( turnLeft i) m)
        #            0    1                                      2
        self.post_insert_locs = {}
        for body in tree_index.all_bodies:
            for i in range(len(body.elems)):
                self.pre_insert_locs[body.elems[i]['span'][1]] = (body.elems,
                                                                  i + 1)
                self.post_insert_locs[body.elems[i]['span'][0]] = (body.elems,
                                                                   i)
            if body.elems:
                self.pre_insert_locs[body.elems[0]['span'][0] - 1] = (
                    body.elems, 0)
                self.post_insert_locs[body.elems[-1]['span'][1] + 1] = (
                    body.elems, len(body.elems))
            else:
                assert body.type == 'run'
                self.pre_insert_locs[body.node['span'][1] - 1] = (body.elems,
                                                                  0)
                self.post_insert_locs[body.node['span'][1]] = (body.elems,
                                                               len(body.elems))

        # ADD_ACTION
        #self.add_action_locs = self.pre_insert_locs
        self.add_action_locs = self.post_insert_locs

        # REMOVE_ACTION
        self.remove_action_locs = {
            body_elems[i]['span'][0]: (body.elems, i)
            for body_elems, i in tree_index.remove_locs
        }

        # REPLACE_ACTION
        self.replace_action_locs = {
            body_elems[i]['span'][0]: (body.elems, i)
            for body_elems, i in tree_index.action_locs
        }

        # UNWRAP_BLOCK
        self.unwrap_block_locs = {
            body_elems[i]['span'][0]: (body.elems, i)
            for body_elems, i in tree_index.unwrappables
        }

        # WRAP_BLOCK
        # block start: pre_insert_locs
        # block end: post_insert_locs

        # WRAP_IFELSE
        # if block start: pre_insert_locs
        # if block end/else block start: post_insert_locs
        # else block end: post_insert_locs

        # REPLACE_COND
        # Not yet implemented

        # SWITCH_IF_WHILE
        # Not yet implemented

    def sample(self):
        raise NotImplementedError

    def contains(self, action):
        action_type, args = action

        if action_type == mutation.ADD_ACTION:
            location, karel_action = args
            return (location in self.add_action_locs and
                    karel_action in mutation.ACTION_NAMES)

        elif action_type == mutation.REMOVE_ACTION:
            location, = args
            return location in self.remove_action_locs

        elif action_type == mutation.REPLACE_ACTION:
            location, karel_action = args
            return (location in self.replace_action_locs and
                    karel_action in mutation.ACTION_NAMES)

        elif action_type == mutation.UNWRAP_BLOCK:
            location, = args
            return location in self.unwrap_block_locs

        elif action_type == mutation.WRAP_BLOCK:
            block_type, cond_id, start, end = args
            if block_type not in ('if', 'while', 'repeat'):
                return False
            if block_type == 'repeat' and not (
                    0 <= cond_id < len(mutation.REPEAT_COUNTS)):
                return False
            if not (0 <= cond_id < len(mutation.CONDS)):
                return False

            start_valid = start in self.pre_insert_locs
            end_valid = end in self.post_insert_locs
            if not (start_valid and end_valid):
                return False
            start_body, start_i = self.pre_insert_locs[start]
            end_body, end_i = self.post_insert_locs[end]
            return start_body is end_body and start_i <= end_i

        elif action_type == mutation.WRAP_IFELSE:
            cond_id, if_start, else_start, end = args
            if not (0 <= cond_id < len(mutation.CONDS)):
                return False

            if_start_valid = if_start in self.pre_insert_locs
            else_start_valid = else_start in self.post_insert_locs
            end_valid = end in self.post_insert_locs

            if not (if_start_valid and else_start_valid and end_valid):
                return False

            if_start_body, if_start_i = self.pre_insert_locs[if_start]
            else_start_body, else_start_i = self.post_insert_locs[else_start]
            end_body, end_i = self.post_insert_locs[end]
            return (if_start_body is else_start_body is end_body and
                    if_start_i <= else_start_i <= end_i)

        elif action_type == mutation.REPLACE_COND:
            # Not yet implemented
            return False
        elif action_type == mutation.SWITCH_IF_WHILE:
            # Not yet implemented
            return False
        else:
            return False


class KarelRefineEnv(gym.Env):
    executor = executor_mod.KarelExecutor()

    def __init__(self, input_tests):
        self.input_tests = input_tests
        self.reset()

    # Overridden methods
    def step(self, action):
        action_type, args = action

        if action_type == mutation.ADD_ACTION:
            location, karel_action = args
            body_elems, i = self.action_space.add_action_locs[location]
            body_elems.insert(i, mutation.ACTIONS_DICT[karel_action])

        elif action_type == mutation.REMOVE_ACTION:
            location, = args
            body_elems, i = self.action_space.remove_action_locs[location]
            del body_elems[i]

        elif action_type == mutation.REPLACE_ACTION:
            location, karel_action = args
            body_elems, i = self.action_space.replace_action_locs[location]
            body_elems[i] = mutation.ACTIONS_DICT[karel_action]

        elif action_type == mutation.UNWRAP_BLOCK:
            location, = args
            body_elems, i = self.unwrap_block_locs[location]
            block = body_elems[i]
            del body_elems[i]
            body_elems[i:i] = block.get('body', [])
            body_elems[i:i] = block.get('elseBody', [])
            body_elems[i:i] = block.get('ifBody', [])

        elif action_type == mutation.WRAP_BLOCK:
            block_type, cond_id, start, end = args
            body_elems, start_i = self.action_space.pre_insert_locs[start]
            _, end_i = self.action_space.post_insert_locs[end]

            subseq = body_elems[start_i:end_i]
            del body_elems[start_i:end_i]
            if block_type == 'repeat':
                new_block = {
                    'type': 'repeat',
                    'times': mutation.REPEAT_COUNTS[cond_id]
                }
            else:
                new_block = {
                    'type': block_type,
                    'cond': mutation.CONDS[cond_id]
                }
            new_block['body'] = subseq
            body_elems.insert(start_i, new_block)

        elif action_type == mutation.WRAP_IFELSE:
            cond_id, if_start, else_start, end = args

            body_elems, if_start_i = self.action_space.pre_insert_locs[
                if_start]
            _, else_start_i = self.action_space.post_insert_locs[else_start]
            _, end_i = self.action_space.post_insert_locs[end]

            if_body = body_elems[if_start_i:else_start_i]
            else_body = body_elems[else_start_i:end_i]
            del body_elems[if_start_i:end]
            new_block = {
                'type': 'ifElse',
                'cond': mutation.CONDS[cond_id],
                'ifBody': if_body,
                'elseBody': else_body
            }
            body_elems.insert(if_start_i, new_block)

        #elif action_type == mutation.REPLACE_COND:
        #   # Not yet implemented
        #    return False #elif action_type == mutation.SWITCH_IF_WHILE:
        #    # Not yet implemented
        #    return False
        else:
            raise ValueError(action_type)

        # Turn tree into tokens
        self.code = parser_for_synthesis.tree_to_tokens(self.tree)

        # Update action space, which will also reparse tokens to get tree
        # containing span information
        self.action_space = MutationActionSpace(code=self.code)

        # Update tree
        self.tree = self.action_space.tree

        # Run new program on I/O grids to get observation
        observation, done = self.compute_obs()
        reward = float(done)
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.code = ['DEF', 'run', 'm(', 'm)']
        self.action_space = MutationActionSpace(code=self.code)
        self.tree = self.action_space.tree
        obs, _ = self.compute_obs()
        return obs

    def compute_obs(self):
        outputs, traces = [], []
        all_correct = True
        for test in self.input_tests:
            result, trace = self.executor.execute(
                self.code, None, test['input'], record_trace=True)
            outputs.append(result)
            traces.append(trace)
            if result != test['output']:
                all_correct = False
        return {
            'code': tuple(self.code),
            'inputs': [test['input'] for test in self.input_tests],
            'outputs': outputs,
            'traces': traces,
            'desired_outputs': [test['output'] for test in self.input_tests],
        }, all_correct


class ComputeAddOps(object):
    # Possible edit operations:
    # - ADD_ACTION
    # - WRAP_BLOCK
    # - WRAP_IFELSE

    conds = [
        'frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent',
        'noMarkersPresent'
    ]
    conds.extend(('not', v) for v in conds[:3])
    tokens = (mutation.ACTION_NAMES + tuple(
        item
        for sublist in ([(('if', cond), ('endIf', cond), ('ifElse', cond), (
            'else', cond), ('endIfElse', cond), ('while', cond
                                                 ), ('endWhile', cond))
                         for cond in conds] + [(('repeat', i), ('endRepeat', i)
                                                ) for i in range(2, 11)])
        for item in sublist))

    token_to_idx = {token: i for i, token in enumerate(tokens)}
    idx_to_token = {i: token for token, i in token_to_idx.items()}

    @classmethod
    def linearize(cls, node):
        if isinstance(node, list):
            return tuple(
                token for item in node for token in cls.linearize(item))

        node_type = node['type']
        if node_type == 'run':
            return cls.linearize(node['body'])
        elif node_type == 'if':
            cond = cls.linearize_cond(node['cond'])
            return ((cls.token_to_idx['if', cond],
                     ) + cls.linearize(node['body']) +
                    (cls.token_to_idx['endIf', cond], ))
        elif node_type == 'ifElse':
            cond = cls.linearize_cond(node['cond'])
            return ((cls.token_to_idx['ifElse', cond],
                     ) + cls.linearize(node['ifBody']) +
                    (cls.token_to_idx['else', cond],
                     ) + cls.linearize(node['elseBody']) +
                    (cls.token_to_idx['endIfElse', cond], ))
        elif node_type == 'while':
            cond = cls.linearize_cond(node['cond'])
            return ((cls.token_to_idx['while', cond],
                     ) + cls.linearize(node['body']) +
                    (cls.token_to_idx['endWhile', cond], ))
        elif node_type == 'repeat':
            return ((cls.token_to_idx['repeat', node['times']['value']],
                     ) + cls.linearize(node['body']) +
                    (cls.token_to_idx['endRepeat', node['times']['value']], ))
        else:
            return (cls.token_to_idx[node_type], )

    @classmethod
    def linearize_cond(cls, node):
        if node['type'] == 'not':
            return ('not', node['cond']['type'])
        else:
            return node['type']

    def __init__(self, goal_tree):
        self.linearized_goal = self.linearize(goal_tree)
        self.goal_actions = None

    def run(self, current_code):
        raise NotImplementedError


def is_subseq(a, b):
    b_it = iter(b)
    return all(a_elem in b_it for a_elem in a)


def subseq_insertions(a, b, debug=False):
    # "not a" doesn't work on NumPy arrays
    if len(a) == 0:
        return [set(b)]

    b_idx = 0
    left_bound = []
    for a_elem in a:
        while a_elem != b[b_idx]:
            b_idx += 1
        left_bound.append(b_idx)
        b_idx += 1

    b_idx = len(b) - 1
    right_bound = []
    for a_elem in reversed(a):
        while a_elem != b[b_idx]:
            b_idx -= 1
        right_bound.append(b_idx)
        b_idx -= 1
    right_bound = right_bound[::-1]

    min_left_bound = min(left_bound)
    b_index = collections.defaultdict(list)
    for i, b_elem in enumerate(b[min_left_bound:max(right_bound) + 1]):
        b_index[b_elem].append(i + min_left_bound)

    result = [set()]
    # Before a[0]
    for b_idx in b_index[a[0]]:
        # For all occurrences of a[0]...
        if b_idx < left_bound[0]:
            continue
        elif b_idx > right_bound[0]:
            break
        # Above: skip if not left_bound[0] <= b_idx <= right_bound[0]
        if b_idx > 0:
            result[0].add(b[b_idx - 1])

    # After a[0], ..., a[-1]
    for i in range(len(a)):
        insert = set()
        for b_idx in b_index[a[i]]:
            # For all occurrences of a[i]...
            if b_idx < left_bound[i]:
                continue
            elif b_idx > right_bound[i]:
                break
            # Above: skip if not left_bound[i] <= b_idx <= right_bound[i]

            # Insert the element to the right of b[b_idx]
            # if it's eligible for insertion
            if b_idx + 1 < len(b) and (
                    i + 1 == len(a) or
                    left_bound[i + 1] != right_bound[i + 1]):
                insert.add(b[b_idx + 1])
        result.append(insert)

    if debug:
        return result, left_bound, right_bound
    return result
