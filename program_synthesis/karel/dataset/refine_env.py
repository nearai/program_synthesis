import collections

from cached_property import cached_property
import gym
import numpy as np

from program_synthesis.karel.dataset import executor as executor_mod
from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.dataset import parser_for_synthesis


class AnnotatedTree(object):
    parser = parser_for_synthesis.KarelForSynthesisParser(build_tree=True)

    def __init__(self, tree=None, code=None):
        assert (tree is None) ^ (code is None)
        if tree:
            self.tree = tree
        else:
            self.tree = self.parser.parse(code)

    def notify_mutation(self):
        self.__dict__.pop('code', None)
        self.__dict__.pop('index', None)
        self.__dict__.pop('pre_insert_locs', None)
        self.__dict__.pop('post_insert_locs', None)
        self.__dict__.pop('remove_action_locs', None)
        self.__dict__.pop('replace_action_locs', None)
        self.__dict__.pop('unwrap_block_locs', None)
        self.__dict__.pop('linearized', None)

        # Tree -> code -> tree so that self.tree has correct spans
        self.tree = self.parser.parse(self.code)

    @cached_property
    def code(self):
        return parser_for_synthesis.tree_to_tokens(self.tree)

    @cached_property
    def index(self):
        return mutation.TreeIndex(self.tree)

    # points to token before insert point
    #           v    v                                      v
    # DEF run m( move IF c( markersPresent c) i( turnLeft i) m)
    #         0  1                                        2
    @cached_property
    def pre_insert_locs(self):
        index = self.index
        pre_insert_locs = {}

        for body in index.all_bodies:
            for i in range(len(body.elems)):
                pre_insert_locs[body.elems[i]['span'][1]] = (body.elems, i + 1)
            if body.elems:
                pre_insert_locs[body.elems[0]['span'][0] - 1] = (body.elems, 0)
            else:
                if body.type in ('run', 'if', 'while', 'repeat'):
                    pre_insert_locs[body.node['span'][1] - 1] = (body.elems, 0)
                elif body.type == 'ifElse-if':
                    pre_insert_locs[body.node['ifSpan'][0]] = (body.elems, 0)
                elif body.type == 'ifElse-else':
                    pre_insert_locs[body.node['elseSpan'][0]] = (body.elems, 0)
                else:
                    raise ValueError(body.type)

        return pre_insert_locs

    # points to token after insert point
    #           v    v                                      v
    # DEF run m( move IF c( markersPresent c) i( turnLeft i) m)
    #            0    1                                      2
    #           v    v    v
    # DEF run m( move move m)
    #            0    1    2
    @cached_property
    def post_insert_locs(self):
        index = self.index
        post_insert_locs = {}

        for body in index.all_bodies:
            for i in range(len(body.elems)):
                post_insert_locs[body.elems[i]['span'][0]] = (body.elems, i)
            if body.elems:
                post_insert_locs[body.elems[-1]['span'][1] + 1] = (
                    body.elems, len(body.elems))
            else:
                if body.type in ('run', 'if', 'while', 'repeat'):
                    post_insert_locs[body.node['span'][1]] = (body.elems,
                                                              len(body.elems))
                elif body.type == 'ifElse-if':
                    post_insert_locs[body.node['ifSpan'][1]] = (
                        body.elems, len(body.elems))
                elif body.type == 'ifElse-else':
                    post_insert_locs[body.node['elseSpan'][1]] = (
                        body.elems, len(body.elems))
                else:
                    raise ValueError(body.type)

        return post_insert_locs

    # ADD_ACTION
    @property
    def add_action_locs(self):
        #return self.pre_insert_locs
        return self.post_insert_locs

    # REMOVE_ACTION
    @cached_property
    def remove_action_locs(self):
        return {
            body_elems[i]['span'][0]: (body_elems, i)
            for body_elems, i in self.index.remove_locs
        }

    # REPLACE_ACTION
    @cached_property
    def replace_action_locs(self):
        return {
            body_elems[i]['span'][0]: (body_elems, i)
            for body_elems, i in self.index.action_locs
        }

    # UNWRAP_BLOCK
    @cached_property
    def unwrap_block_locs(self):
        return {
            body_elems[i]['span'][0]: (body_elems, i)
            for body_elems, i in self.index.unwrappables
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

    @cached_property
    def linearized(self):
        return ComputeAddOps.linearize(self.tree)


class MutationActionSpace(gym.Space):
    def __init__(self, tree=None, code=None, atree=None):
        if atree is None:
            self.atree = AnnotatedTree(tree, code)
        else:
            self.atree = atree

    def sample(self):
        raise NotImplementedError

    def contains(self, action):
        action_type, args = action

        if action_type == mutation.ADD_ACTION:
            location, karel_action = args
            return (location in self.atree.add_action_locs and
                    karel_action in mutation.ACTION_NAMES)

        elif action_type == mutation.REMOVE_ACTION:
            location, = args
            return location in self.atree.remove_action_locs

        elif action_type == mutation.REPLACE_ACTION:
            location, karel_action = args
            return (location in self.atree.replace_action_locs and
                    karel_action in mutation.ACTION_NAMES)

        elif action_type == mutation.UNWRAP_BLOCK:
            location, = args
            return location in self.atree.unwrap_block_locs

        elif action_type == mutation.WRAP_BLOCK:
            block_type, cond_id, start, end = args
            if block_type not in ('if', 'while', 'repeat'):
                return False
            if block_type == 'repeat':
                if not 0 <= cond_id < len(mutation.REPEAT_COUNTS):
                    return False
            else:
                if not 0 <= cond_id < len(mutation.CONDS):
                    return False

            start_valid = start in self.atree.pre_insert_locs
            end_valid = end in self.atree.post_insert_locs
            if not (start_valid and end_valid):
                return False
            start_body, start_i = self.atree.pre_insert_locs[start]
            end_body, end_i = self.atree.post_insert_locs[end]
            return start_body is end_body and start_i <= end_i

        elif action_type == mutation.WRAP_IFELSE:
            cond_id, if_start, else_start, end = args
            if not (0 <= cond_id < len(mutation.CONDS)):
                return False

            if_start_valid = if_start in self.atree.pre_insert_locs
            else_start_valid = else_start in self.atree.post_insert_locs
            end_valid = end in self.atree.post_insert_locs

            if not (if_start_valid and else_start_valid and end_valid):
                return False

            if_start_body, if_start_i = self.atree.pre_insert_locs[if_start]
            else_start_body, else_start_i = self.atree.post_insert_locs[
                else_start]
            end_body, end_i = self.atree.post_insert_locs[end]
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

    def apply(self, action):
        action_type, args = action

        if action_type == mutation.ADD_ACTION:
            location, karel_action = args
            body_elems, i = self.atree.add_action_locs[location]
            body_elems.insert(i, mutation.ACTIONS_DICT[karel_action])

        elif action_type == mutation.REMOVE_ACTION:
            location, = args
            body_elems, i = self.atree.remove_action_locs[location]
            del body_elems[i]

        elif action_type == mutation.REPLACE_ACTION:
            location, karel_action = args
            body_elems, i = self.atree.replace_action_locs[location]
            body_elems[i] = mutation.ACTIONS_DICT[karel_action]

        elif action_type == mutation.UNWRAP_BLOCK:
            location, = args
            body_elems, i = self.atree.unwrap_block_locs[location]
            block = body_elems[i]
            del body_elems[i]
            body_elems[i:i] = block.get('body', [])
            body_elems[i:i] = block.get('elseBody', [])
            body_elems[i:i] = block.get('ifBody', [])

        elif action_type == mutation.WRAP_BLOCK:
            block_type, cond_id, start, end = args
            body_elems, start_i = self.atree.pre_insert_locs[start]
            _, end_i = self.atree.post_insert_locs[end]

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

            body_elems, if_start_i = self.atree.pre_insert_locs[if_start]
            _, else_start_i = self.atree.post_insert_locs[else_start]
            _, end_i = self.atree.post_insert_locs[end]

            if_body = body_elems[if_start_i:else_start_i]
            else_body = body_elems[else_start_i:end_i]
            del body_elems[if_start_i:end_i]
            new_block = {
                'type': 'ifElse',
                'cond': mutation.CONDS[cond_id],
                'ifBody': if_body,
                'elseBody': else_body
            }
            body_elems.insert(if_start_i, new_block)

        #elif action_type == mutation.REPLACE_COND:
        #   # Not yet implemented
        #    return False 
        #elif action_type == mutation.SWITCH_IF_WHILE:
        #    # Not yet implemented
        #    return False
        else:
            raise ValueError(action_type)

        self.atree.notify_mutation()

    def enumerate_additive_actions(self):
        for location in self.add_action_locs:
            for karel_action in mutation.ACTION_NAMES:
                yield mutation.ADD_ACTION, (location, karel_action)

        for loc1, (body1, i1) in self.atree.pre_insert_locs.iteritems():
            for loc2, (body2, i2) in self.atree.post_insert_locs.iteritems():
                if body2 is not body1 or i2 < i1:
                    continue

                for r in range(len(mutation.REPEAT_COUNTS)):
                    yield mutation.WRAP_BLOCK, ('repeat', r, loc1, loc2)

                for cond_id in range(len(mutation.CONDS)):
                    yield mutation.WRAP_BLOCK, ('if', cond_id, loc1, loc2)
                    yield mutation.WRAP_BLOCK, ('while', cond_id, loc1, loc2)
                    for loc3, (body3,
                               i3) in self.atree.post_insert_locs.iteritems():
                        if body3 is not body2 or i3 < i2:
                            continue
                        yield mutation.WRAP_IFELSE, (cond_id, loc1, loc2, loc3)


class KarelRefineEnv(gym.Env):
    executor = executor_mod.KarelExecutor()

    def __init__(self, input_tests):
        self.input_tests = input_tests
        self.reset()

    # Overridden methods
    def step(self, action):
        self.action_space.apply(action)

        # Run new program on I/O grids to get observation
        observation, done = self.compute_obs()
        reward = float(done)
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.reset_with(('DEF', 'run', 'm(', 'm)'))
        obs, _ = self.compute_obs()
        return obs

    def reset_with(self, code):
        self.action_space = MutationActionSpace(code=code)
        self.atree = self.action_space.atree

    def compute_obs(self):
        outputs, traces = [], []
        all_correct = True
        for test in self.input_tests:
            result, trace = self.executor.execute(
                self.atree.code, None, test['input'], record_trace=True)
            outputs.append(result)
            traces.append(trace)
            if result != test['output']:
                all_correct = False
        return {
            'code': tuple(self.atree.code),
            'inputs': [test['input'] for test in self.input_tests],
            'outputs': outputs,
            'traces': traces,
            'desired_outputs': [test['output'] for test in self.input_tests],
        }, all_correct


def linearize_cond(node):
    if node['type'] == 'not':
        return ('not', node['cond']['type']), 4
    else:
        return node['type'], 1


class ComputeAddOps(object):
    conds = [
        'frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent',
        'noMarkersPresent'
    ]
    conds.extend(('not', v) for v in conds[:3])
    tokens = (mutation.ACTION_NAMES + tuple(
        item
        for sublist in ([(('if', cond), ('end-if', cond), ('ifElse', cond), (
            'else', cond), ('end-ifElse', cond), ('while', cond
                                                  ), ('end-while', cond))
                         for cond in conds] + [(('repeat', i), ('end-repeat', i
                                                                ))
                                               for i in range(2, 11)])
        for item in sublist))

    token_to_idx = {token: i for i, token in enumerate(tokens)}
    idx_to_token = {i: token for token, i in token_to_idx.items()}

    action_ids = set()
    for t in mutation.ACTION_NAMES:
        action_ids.add(token_to_idx[t])
    cond_to_id = {
        linearize_cond(c)[0]: i
        for i, c in enumerate(mutation.CONDS)
    }
    repeat_count_to_id = {
        r['value']: i
        for i, r in enumerate(mutation.REPEAT_COUNTS)
    }

    @classmethod
    def linearize(cls, node, offset=0):
        if isinstance(node, list):
            tokens = []
            orig_spans = []
            for item in node:
                new_tokens, new_spans = cls.linearize(item, offset)
                tokens += new_tokens
                orig_spans += new_spans
                offset = new_spans[-1][1] + 1
            return tuple(tokens), tuple(orig_spans)

        node_type = node['type']
        if node_type == 'run':
            return cls.linearize(node['body'], offset + 3)

        elif node_type == 'ifElse':
            cond, cond_length = linearize_cond(node['cond'])
            if_body_tokens, if_body_spans = cls.linearize(
                node['ifBody'],
                # +4: tokens in IFELSE c( c) i(
                offset + cond_length + 4)
            begin_span = (offset, offset + cond_length + 3)
            if if_body_spans:
                mid_span_left = if_body_spans[-1][1] + 1
            else:
                mid_span_left = begin_span[1] + 1
            # +2: from i) ELSE i(
            mid_span = (mid_span_left, mid_span_left + 2)
            else_body_tokens, else_body_spans = cls.linearize(node['elseBody'],
                                                              mid_span[1] + 1)
            if else_body_spans:
                end_index = else_body_spans[-1][1] + 1
            else:
                end_index = mid_span[1] + 1
            end_span = (end_index, end_index)

            tokens = ((cls.token_to_idx['ifElse', cond], ) + if_body_tokens +
                      (cls.token_to_idx['else', cond], ) + else_body_tokens +
                      (cls.token_to_idx['end-ifElse', cond], ))
            orig_spans = (begin_span, ) + if_body_spans + (
                mid_span, ) + else_body_spans + (end_span, )
            return tokens, orig_spans

        elif node_type in ('if', 'while', 'repeat'):
            if node_type == 'repeat':
                cond = node['times']['value']
                # +3: tokens in REPEAT R=? w(
                body_offset = offset + 3
            else:
                cond, cond_length = linearize_cond(node['cond'])
                # +4: tokens in IF c( c) i(
                #               WHILE c( c) w(
                body_offset = offset + cond_length + 4

            body_tokens, body_spans = cls.linearize(node['body'], body_offset)
            begin_span = (offset, body_offset - 1)
            # i), w), r)
            if body_spans:
                end_index = body_spans[-1][1] + 1
            else:
                # IF c( ... c) i( i)
                #                 ^
                #            body_offset
                end_index = body_offset
            end_span = (end_index, end_index)

            tokens = (cls.token_to_idx[node_type, cond], ) + body_tokens + (
                cls.token_to_idx['end-' + node_type, cond], )
            orig_spans = (begin_span, ) + body_spans + (end_span, )
            return tokens, orig_spans

        else:
            return (cls.token_to_idx[node_type], ), ((offset, offset), )

    def __init__(self, goal_tree):
        self.goal_atree = AnnotatedTree(tree=goal_tree)
        self.goal, _ = self.goal_atree.linearized

    @classmethod
    def run(cls, cur_atree=None, goal_atree=None):
        linearized, spans = cur_atree.linearized
        goal = goal_atree.linearized[0]

        # Example:
        #       move move
        # pos: 0    1    2
        # span: 3,3  4,4
        def pos_to_pre_insert_loc(pos):
            if not spans:
                #         v
                # DEF run m( m)
                # 0   1   2  3
                return 2
            if pos == 0:
                return spans[pos][0] - 1
            else:
                return spans[pos - 1][1]

        def pos_to_post_insert_loc(pos):
            if not spans:
                #            v
                # DEF run m( m)
                # 0   1   2  3
                return 3
            if pos == len(spans):
                return spans[pos - 1][1] + 1
            else:
                return spans[pos][0]

        def insert(seq, item, pos):
            return seq[:pos] + (item, ) + seq[pos:]

        result = []
        insertions = subseq_insertions(linearized, goal)
        for pos, items in enumerate(insertions):
            for item in items:
                if item in cls.action_ids:
                    if not subseq_valid_remainder_exists(
                            insert(linearized, item, pos), goal,
                            CheckBlocksWellFormed()):
                        continue
                    result.append((mutation.ADD_ACTION, (
                        pos_to_post_insert_loc(pos), cls.idx_to_token[item])))
                    continue

                # Must be if, ifElse, while, or repeat
                # or the end- counterparts
                token_type, cond = cls.idx_to_token[item]
                if token_type[:4] in ('end-', 'else'):
                    continue

                # Try starting the block here
                block_started = insert(linearized, item, pos)
                start_loc = pos_to_pre_insert_loc(pos)
                if token_type == 'ifElse':
                    else_token = cls.token_to_idx['else', cond]
                    end_token = cls.token_to_idx['end-ifElse', cond]
                    cond_id = cls.cond_to_id[cond]

                    else_insertions = subseq_insertions(block_started,
                                                        goal)
                    for else_pos in range(pos + 1, len(block_started) + 1):
                        if else_token not in else_insertions[else_pos]:
                            continue
                        else_loc = pos_to_post_insert_loc(else_pos - 1)
                        if (cur_atree.pre_insert_locs[start_loc][0] is not
                                cur_atree.post_insert_locs[else_loc][0]):
                            continue
                        else_inserted = insert(block_started, else_token,
                                               else_pos)
                        end_insertions = subseq_insertions(else_inserted,
                                                           goal)
                        for end_pos in range(else_pos + 1,
                                             len(else_inserted) + 1):
                            if end_token not in end_insertions[end_pos]:
                                continue
                            end_loc = pos_to_post_insert_loc(end_pos - 2)
                            if (cur_atree.post_insert_locs[else_loc][0] is not
                                   cur_atree.post_insert_locs[end_loc][0]):
                                continue
                            end_inserted = insert(else_inserted, end_token,
                                                  end_pos)
                            if not subseq_valid_remainder_exists(
                                    end_inserted, goal,
                                    CheckBlocksWellFormed()):
                                continue
                            result.append((
                                mutation.WRAP_IFELSE,
                                (
                                    cond_id,
                                    # ifElse
                                    start_loc,
                                    # else
                                    else_loc,
                                    # end
                                    end_loc)))
                else:
                    end_token = cls.token_to_idx['end-' + token_type, cond]
                    end_insertions = subseq_insertions(block_started,
                                                       goal)
                    cond_id = (cls.repeat_count_to_id[cond]
                               if token_type == 'repeat' else
                               cls.cond_to_id[cond])
                    for end_pos in range(pos + 1, len(block_started) + 1):
                        if end_token not in end_insertions[end_pos]:
                            continue
                        end_loc = pos_to_post_insert_loc(end_pos - 1)
                        if (cur_atree.pre_insert_locs[start_loc][0] is not
                               cur_atree.post_insert_locs[end_loc][0]):
                            continue
                        end_inserted = insert(block_started, end_token,
                                              end_pos)
                        if not subseq_valid_remainder_exists(
                                end_inserted, goal,
                                CheckBlocksWellFormed()):
                            continue
                        result.append((
                            mutation.WRAP_BLOCK,
                            (
                                token_type,
                                cond_id,
                                # start
                                start_loc,
                                # end
                                end_loc)))

        return result


class CheckBlocksWellFormed(object):
    # if/end-if, while/end-while, repeat/end-repeat
    all_pairs = [(ComputeAddOps.token_to_idx[block_type, cond],
                  ComputeAddOps.token_to_idx['end-' + block_type, cond])
                 for cond in ComputeAddOps.conds
                 for block_type in ('if', 'while')]
    all_pairs += [(ComputeAddOps.token_to_idx['repeat', i],
                   ComputeAddOps.token_to_idx['end-repeat', i])
                  for i in range(2, 11)]

    # ifElse, else, end-ifElse
    all_triples = [(ComputeAddOps.token_to_idx['ifElse', cond],
                    ComputeAddOps.token_to_idx['else', cond],
                    ComputeAddOps.token_to_idx['end-ifElse', cond])
                   for cond in ComputeAddOps.conds]

    block_delimiters = {}
    for a, b in all_pairs:
        # (what to expect from top of stack after popping,
        #  what to push onto stack)
        block_delimiters[a] = (None, a)
        block_delimiters[b] = (a, None)
    for a, b, c in all_triples:
        block_delimiters[a] = (None, a)
        block_delimiters[b] = (a, b)
        block_delimiters[c] = (b, None)

    __slots__ = ('stack', 'remainder_stack', 'index')

    def __init__(self, stack=None, remainder_stack=None, index=None):
        if stack is None:
            stack = []
            remainder_stack = []
            index = 0
        self.stack = stack
        self.remainder_stack = remainder_stack
        self.index = index

    def __call__(self, superseq_prefix, subseq_tokens):
        stack = self.stack
        remainder_stack = self.remainder_stack

        # Update stack with tokens from superseq_prefix
        for token in superseq_prefix:
            stack_actions = self.block_delimiters.get(token)
            if stack_actions is not None:
                stack_pop, stack_push = stack_actions
                if stack_pop is not None:
                    if not self.stack:
                        # Encountering end-if before if, etc.
                        return None
                    if stack is self.stack: stack = list(self.stack)
                    stack_top, _  = stack.pop()
                    if stack_top != stack_pop:
                        # Encountering if then end-while, etc
                        return None
                if stack_push is not None:
                    if stack is self.stack: stack = list(self.stack)
                    stack.append((stack_push, self.index))
            self.index += 1

        # If subseq_token is a block delimiter, update stack 
        for token in subseq_tokens:
            stack_actions = self.block_delimiters.get(token)
            if stack_actions is not None:
                stack_pop, stack_push = stack_actions
                if stack_pop is not None:
                    if not self.stack or not self.remainder_stack:
                        # Encountering end-if before if, etc.
                        return None
                    # Check that remainder_stack's top is the same as the
                    # current stack's top
                    if self.stack[-1] != self.remainder_stack[-1]:
                        return None
                    if stack is self.stack: stack = list(self.stack)
                    stack_top, _ = stack.pop()
                    if stack_top != stack_pop:
                        return None
                    if remainder_stack is self.remainder_stack:
                        remainder_stack = list(self.remainder_stack)
                    remainder_stack.pop()
                if stack_push is not None:
                    if stack is self.stack: stack = list(self.stack)
                    stack.append((stack_push, self.index))
                    if remainder_stack is self.remainder_stack:
                        remainder_stack = list(self.remainder_stack)
                    remainder_stack.append(stack[-1])
            self.index += 1
            
        return CheckBlocksWellFormed(
                stack,
                remainder_stack,
                self.index)


def is_subseq(a, b):
    b_it = iter(b)
    return all(a_elem in b_it for a_elem in a)


def subseq_bounds(a, b):
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

    return left_bound, right_bound


def subseq_insertions(a, b, debug=False):
    # `a` is the subsequence, `b` is the longer sequence.

    # "not a" doesn't work on NumPy arrays
    if len(a) == 0:
        return [set(b)]

    left_bound, right_bound = subseq_bounds(a, b)

    result = [set()]
    # Before a[0]
    result[0].update(b[0:right_bound[0]])
    # After a[0], ..., a[-1]
    for i in range(len(a)):
        insert = set(b[left_bound[i] + 1:right_bound[i + 1]
                       if i + 1 < len(a) else None])
        result.append(insert)

    if debug:
        return result, left_bound, right_bound
    return result


def subseq_valid_remainder_exists(a, b, checker):
    # `a` is the subsequence, `b` is the longer sequence.
    if not a:
        return bool(checker(b, []))

    left_bound, right_bound = subseq_bounds(a, b)

    valid_exists = False
    for i in range(left_bound[0], right_bound[0] + 1):
        if a[0] != b[i]:
            continue

        # Try placing a[0] in b[i]
        new_checker = checker(b[:i], a[0:1])
        if new_checker is None:
            # The current prefix of b is not valid.
            continue

        if subseq_valid_remainder_exists(a[1:], b[i + 1:], new_checker):
            valid_exists = True
            break

    return valid_exists
