from __future__ import print_function
import functools
import six

import ply.lex

from program_synthesis.karel.dataset.parser_base import parser_prompt
from program_synthesis.karel.dataset.parser_base import Parser
from program_synthesis.karel.dataset.utils import KarelSyntaxError


def make_token(type, value=None):
    def f(lexpos):
        t = ply.lex.LexToken()
        t.type = type
        if value is None:
            t.value = type
        else:
            t.value = value
        t.lineno = 0
        t.lexpos = lexpos
        return t
    return f


class KarelForSynthesisParser(Parser):

    tokens = [
            'DEF', 'RUN',
            'M_LBRACE', 'M_RBRACE', 'C_LBRACE', 'C_RBRACE', 'R_LBRACE', 'R_RBRACE',
            'W_LBRACE', 'W_RBRACE', 'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
            'INT', #'NEWLINE', 'SEMI',
            'WHILE', 'REPEAT',
            'IF', 'IFELSE', 'ELSE',
            'FRONT_IS_CLEAR', 'LEFT_IS_CLEAR', 'RIGHT_IS_CLEAR',
            'MARKERS_PRESENT', 'NO_MARKERS_PRESENT', 'NOT',
            'MOVE', 'TURN_RIGHT', 'TURN_LEFT',
            'PICK_MARKER', 'PUT_MARKER',
    ]

    t_ignore =' \t\n'

    t_M_LBRACE = 'm\('
    t_M_RBRACE = 'm\)'

    t_C_LBRACE = 'c\('
    t_C_RBRACE = 'c\)'

    t_R_LBRACE = 'r\('
    t_R_RBRACE = 'r\)'

    t_W_LBRACE = 'w\('
    t_W_RBRACE = 'w\)'

    t_I_LBRACE = 'i\('
    t_I_RBRACE = 'i\)'

    t_E_LBRACE = 'e\('
    t_E_RBRACE = 'e\)'

    t_DEF = 'DEF'
    t_RUN = 'run'
    t_WHILE = 'WHILE'
    t_REPEAT = 'REPEAT'
    t_IF = 'IF'
    t_IFELSE = 'IFELSE'
    t_ELSE = 'ELSE'
    t_NOT = 'not'

    t_FRONT_IS_CLEAR = 'frontIsClear'
    t_LEFT_IS_CLEAR = 'leftIsClear'
    t_RIGHT_IS_CLEAR = 'rightIsClear'
    t_MARKERS_PRESENT = 'markersPresent'
    t_NO_MARKERS_PRESENT = 'noMarkersPresent'

    conditional_functions = [
            t_FRONT_IS_CLEAR, t_LEFT_IS_CLEAR, t_RIGHT_IS_CLEAR,
            t_MARKERS_PRESENT, t_NO_MARKERS_PRESENT,
    ]

    t_MOVE = 'move'
    t_TURN_RIGHT = 'turnRight'
    t_TURN_LEFT = 'turnLeft'
    t_PICK_MARKER = 'pickMarker'
    t_PUT_MARKER = 'putMarker'

    action_functions = [
            t_MOVE,
            t_TURN_RIGHT, t_TURN_LEFT,
            t_PICK_MARKER, t_PUT_MARKER,
    ]

    string_to_token_map = {
            'DEF': make_token('DEF'),
            'run': make_token('RUN', 'run'),
            'm(': make_token('M_LBRACE', 'm('),
            'm)': make_token('M_RBRACE', 'm)'),
            'c(': make_token('C_LBRACE', 'c('),
            'c)': make_token('C_RBRACE', 'c)'),
            'r(': make_token('R_LBRACE', 'r('),
            'r)': make_token('R_RBRACE', 'r)'),
            'w(': make_token('W_LBRACE', 'w('),
            'w)': make_token('W_RBRACE', 'w)'),
            'i(': make_token('I_LBRACE', 'i('),
            'i)': make_token('I_RBRACE', 'i)'),
            'e(': make_token('E_LBRACE', 'e('),
            'e)': make_token('E_RBRACE', 'e)'),
            'R=2': make_token('INT', 2),
            'R=3': make_token('INT', 3),
            'R=4': make_token('INT', 4),
            'R=5': make_token('INT', 5),
            'R=6': make_token('INT', 6),
            'R=7': make_token('INT', 7),
            'R=8': make_token('INT', 8),
            'R=9': make_token('INT', 9),
            'R=10': make_token('INT', 10),
            'WHILE': make_token('WHILE'),
            'REPEAT': make_token('REPEAT'),
            'IF': make_token('IF'),
            'IFELSE': make_token('IFELSE'),
            'ELSE': make_token('ELSE'),
            'frontIsClear': make_token('FRONT_IS_CLEAR', 'frontIsClear'),
            'leftIsClear': make_token('LEFT_IS_CLEAR', 'leftIsClear'),
            'rightIsClear': make_token('RIGHT_IS_CLEAR','rightIsClear'),
            'markersPresent': make_token('MARKERS_PRESENT', 'markersPresent'),
            'noMarkersPresent': make_token('NO_MARKERS_PRESENT', 'noMarkersPresent'),
            'not': make_token('NOT', 'not'),
            'move': make_token('MOVE', 'move'),
            'turnRight': make_token('TURN_RIGHT', 'turnRight'),
            'turnLeft': make_token('TURN_LEFT', 'turnLeft'),
            'pickMarker': make_token('PICK_MARKER', 'pickMarker'),
            'putMarker': make_token('PUT_MARKER', 'putMarker'),
    }

    #########
    # lexer
    #########

    INT_PREFIX = 'R='
    def t_INT(self, t):
        r'R=\d+'

        value = int(t.value.replace(self.INT_PREFIX, ''))
        if not (self.min_int <= value <= self.max_int):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`". \
                    format(self.min_int, self.max_int, value))

        t.value = value
        return t

    def random_INT(self):
        return "{}{}".format(
                self.INT_PREFIX,
                self.rng.randint(self.min_int, self.max_int + 1))

    def t_error(self, t):
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)

    def token_list_to_tokenfunc(self, tokens):
        try:
            tokens = [
                self.string_to_token_map[token](i)
                for i, token in enumerate(tokens)
            ]
        except KeyError as e:
            raise KarelSyntaxError('Unknown token: {}'.format(e))
        tokens.append(None)
        if six.PY2:
            return iter(tokens).next
        return iter(tokens).__next__

    #########
    # parser
    #########

    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt_or_empty M_RBRACE'''
        stmt = p[4]
        if self.build_tree:
            span = (p.lexpos(1), p.lexpos(5))
            prog = {'type': 'run', 'body':  stmt, 'span': span}
        else:
            prog = stmt

        p[0] = prog

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        p[0] = p[1]
        if self.build_tree and not isinstance(p[0], list):
            p[0] = [p[0]]

    def p_stmt_or_empty(self, p):
        '''stmt_or_empty : stmt
                         | empty
        '''
        if p[1] is None:
            if  self.build_tree:
                p[0] = []
            else:
                p[0] = lambda: None
        else:
            p[0] = p[1]

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]

        if self.build_tree:
            stmt_stmt = stmt1 + stmt2
        else:
            def stmt_stmt():
                stmt1()
                stmt2()
        p[0] = stmt_stmt

    def p_if(self, p):
        '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt_or_empty I_RBRACE
        '''
        cond, stmt = p[3], p[6]
        span = (p.lexpos(1), p.lexpos(7))
        true_span = (p.lexpos(5), p.lexpos(7))
        false_span = (p.lexpos(7), p.lexpos(7))

        self.cond_block_spans.append(span)

        if self.build_tree:
            if_ = {'type': 'if', 'cond': cond, 'body': stmt, 'span': span}
        else:
            cond_fn, cond_span = cond
            def if_():
                cond_value = cond_fn()
                self.karel.event_callback('if', span, cond_span, cond_value,
                        true_span if cond_value else false_span)
                if cond_value:
                    stmt()
        p[0] = if_

    def p_ifelse(self, p):
        '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt_or_empty I_RBRACE ELSE E_LBRACE stmt_or_empty E_RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]
        span = (p.lexpos(1), p.lexpos(11))
        true_span = (p.lexpos(5), p.lexpos(7))
        false_span = (p.lexpos(9), p.lexpos(11))

        self.cond_block_spans.append(span)

        if self.build_tree:
            ifelse = {
                'type': 'ifElse',
                'cond': cond,
                'ifBody': stmt1,
                'elseBody': stmt2,
                'span': span,
                'ifSpan': true_span,
                'elseSpan': false_span,
            }
        else:
            cond_fn, cond_span = cond
            def ifelse():
                cond_value = cond_fn()
                self.karel.event_callback('ifElse', span, cond_span, cond_value,
                        true_span if cond_value else false_span)
                if cond_value:
                    stmt1()
                else:
                    stmt2()

        p[0] = ifelse

    def p_while(self, p):
        '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt_or_empty W_RBRACE
        '''
        cond, stmt = p[3], p[6]
        span = (p.lexpos(1), p.lexpos(7))
        true_span = (p.lexpos(5), p.lexpos(7))
        false_span = (p.lexpos(7), p.lexpos(7))

        self.cond_block_spans.append(span)

        if self.build_tree:
            while_ = {
                'type': 'while',
                'cond': cond,
                'body': stmt,
                'span': span,
            }
        else:
            cond_fn, cond_span = cond
            def while_():
                while True:
                    cond_value = cond_fn()
                    self.karel.event_callback('while', span, cond_span, cond_value,
                            true_span if cond_value else false_span)
                    if not cond_value:
                        break
                    stmt()
        p[0] = while_

    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt_or_empty R_RBRACE
        '''
        cste, stmt = p[2], p[4]
        span = (p.lexpos(1), p.lexpos(5))

        if self.build_tree:
            repeat = {
                'type': 'repeat',
                'times': cste,
                'body': stmt,
                'span': span,
            }
        else:
            limit = cste()
            true_span = (p.lexpos(3), p.lexpos(5))
            false_span = (p.lexpos(5), p.lexpos(5))

            def repeat():
                for i in range(limit, 0, -1):
                    self.karel.event_callback('repeat', span, cste.span, i,
                                              true_span)
                    stmt()
                self.karel.event_callback('repeat', span, cste.span, 0,
                        false_span)
        p[0] = repeat

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT C_LBRACE cond_without_not C_RBRACE
        '''
        if p[1] != 'not':
            p[0] = p[1]
            return

        span = (p.lexpos(1), p.lexpos(4))
        if self.build_tree:
            fn = {
                'type': 'not',
                'cond': p[3],
                'span': span,
            }
        else:
            cond_without_not, _ = p[3]
            fn = (lambda: not cond_without_not(), span)

        p[0] = fn

    def p_cond_without_not(self, p):
        '''cond_without_not : FRONT_IS_CLEAR
                            | LEFT_IS_CLEAR
                            | RIGHT_IS_CLEAR
                            | MARKERS_PRESENT
                            | NO_MARKERS_PRESENT
        '''
        cond_without_not = p[1]
        span = (p.lexpos(1), p.lexpos(1))
        if self.build_tree:
            cond = {
                'type': cond_without_not,
                'span': span,
            }
        else:
            cond = (getattr(self.karel, cond_without_not), span)
        p[0] = cond

    def p_action(self, p):
        '''action : MOVE
                  | TURN_RIGHT
                  | TURN_LEFT
                  | PICK_MARKER
                  | PUT_MARKER
        '''
        action_name = p[1]
        span = (p.lexpos(1), p.lexpos(1))
        self.action_spans.append(span)

        if self.build_tree:
            action = {'type': action_name, 'span': span}
        else:
            action = functools.partial(
                    getattr(self.karel, action_name),
                    metadata=span)
        p[0] = action

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        span = (p.lexpos(1), p.lexpos(1))
        if self.build_tree:
            fn = {
                'type': 'count',
                'value': value,
            }
        else:
            value = int(value)
            def fn():
                return value
            fn.span = span
        p[0] = fn

    def p_empty(self, p):
        '''empty :'''
        pass

    def p_error(self, p):
        if p:
            raise KarelSyntaxError("Syntax error at '%s'" % p.value)
        else:
            raise KarelSyntaxError("Syntax error at EOF")


type_to_list_fn = {
        'run': lambda v: ('DEF', 'run', 'm(') +  tree_to_tokens(v['body']) + ('m)',),
        'if': lambda v: ('IF', 'c(') + tree_to_tokens(v['cond']) + ('c)', 'i(')
        + tree_to_tokens(v['body']) + ('i)',),
        'ifElse': lambda v: ('IFELSE', 'c(') + tree_to_tokens(v['cond']) +
        ('c)', 'i(') + tree_to_tokens(v['ifBody']) + ('i)', 'ELSE', 'e(') +
        tree_to_tokens(v['elseBody']) + ('e)',),
        'while': lambda v: ('WHILE', 'c(') + tree_to_tokens(v['cond']) + ('c)', 'w(')
        + tree_to_tokens(v['body']) + ('w)',),
        'repeat': lambda v: ('REPEAT',) + tree_to_tokens(v['times']) + ('r(',)
        + tree_to_tokens(v['body']) + ('r)',),
        'count': lambda v: ('R={:d}'.format(v['value']),),
        'not': lambda v: ('not', 'c(') + tree_to_tokens(v['cond']) + ('c)',),
}
for k in (KarelForSynthesisParser.conditional_functions +
        KarelForSynthesisParser.action_functions):
    type_to_list_fn[k] = lambda v, k=k: (k,)


def tree_to_tokens(node):
    if isinstance(node, list):
        return tuple(token for item in node for token in tree_to_tokens(item))
    return type_to_list_fn[node['type']](node)


if __name__ == '__main__':
    parser = KarelForSynthesisParser()
    parser_prompt(parser)
