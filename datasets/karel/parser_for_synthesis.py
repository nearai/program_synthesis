from __future__ import print_function

import ply.lex

from .parser_base import dummy, get_hash, parser_prompt, Parser
from .utils import KarelSyntaxError


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
        tokens = [
            self.string_to_token_map[token](i)
            for i, token in enumerate(tokens)
        ]
        tokens.append(None)
        return iter(tokens).next

    #########
    # parser
    #########

    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]

        @self.callout
        def fn():
            return stmt()
        fn.tree = {'run': stmt.tree}

        p[0] = fn

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        p[0] = p[1]

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]

        @self.callout
        def fn():
            stmt1() ;stmt2()

        if isinstance(stmt2.tree, list):
            fn.tree =  [stmt1.tree] + stmt2.tree
        else:
            fn.tree = [stmt1.tree, stmt2.tree]
        p[0] = fn

    def p_if(self, p):
        '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
        '''
        cond, stmt = p[3], p[6]

        hit_info = self.hit_info
        if hit_info is not None:
            num = get_hash()
            hit_info[num] += 1

            @self.callout
            def fn():
                if cond():
                    hit_info[num] -= 1
                    out = stmt()
                else:
                    out = dummy()
                return out
        else:
            @self.callout
            def fn():
                if cond():
                    out = stmt()
                else:
                    out = dummy()
                return out

        fn.tree = {'type': 'if', 'cond': cond.tree, 'body': stmt.tree}
        p[0] = fn

    def p_ifelse(self, p):
        '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]

        hit_info = self.hit_info
        if hit_info is not None:
            num1, num2 = get_hash(), get_hash()
            hit_info[num1] += 1
            hit_info[num2] += 1

            @self.callout
            def fn():
                if cond():
                    hit_info[num1] -= 1
                    out = stmt1()
                else:
                    hit_info[num2] -= 1
                    out = stmt2()
                return out
        else:
            @self.callout
            def fn():
                if cond():
                    out = stmt1()
                else:
                    out = stmt2()
                return out

        fn.tree = {'type': 'ifElse', 'cond': cond.tree, 'ifBody': stmt1.tree,
                'elseBody': stmt2.tree}
        p[0] = fn

    def p_while(self, p):
        '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE
        '''
        cond, stmt = p[3], p[6]

        hit_info = self.hit_info
        if hit_info is not None:
            num = get_hash()
            hit_info[num] += 1

            @self.callout
            def fn():
                while(cond()):
                    hit_info[num] -= 1
                    stmt()
        else:
            @self.callout
            def fn():
                while(cond()):
                    stmt()
        fn.tree = {'type': 'while', 'cond': cond.tree, 'body': stmt.tree,
                'span': (p.lexpos(1), p.lexpos(7) + 1)}
        p[0] = fn

    def p_repeat(self, p):
        '''repeat : REPEAT cste R_LBRACE stmt R_RBRACE
        '''
        cste, stmt = p[2], p[4]

        hit_info = self.hit_info
        if hit_info is not None:
            num = get_hash()
            hit_info[num] += 1

            @self.callout
            def fn():
                for _ in range(cste()):
                    hit_info[num] -= 1
                    stmt()
        else:
            @self.callout
            def fn():
                for _ in range(cste()):
                    stmt()

        fn.tree = {'type': 'repeat', 'times': cste.tree, 'body': stmt.tree,
                'span': (p.lexpos(1), p.lexpos(5))}
        p[0] = fn

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT C_LBRACE cond_without_not C_RBRACE
        '''
        if callable(p[1]):
            p[0] = p[1]
        else: # NOT
            cond_without_not = p[3]
            fn = lambda: not cond_without_not()
            fn.tree = {'type': 'not', 'cond': cond_without_not.tree, 'span':
                    (p.lexpos(1), p.lexpos(4) + 1)}
            p[0] = fn

    def p_cond_without_not(self, p):
        '''cond_without_not : FRONT_IS_CLEAR
                            | LEFT_IS_CLEAR
                            | RIGHT_IS_CLEAR
                            | MARKERS_PRESENT
                            | NO_MARKERS_PRESENT
        '''
        cond_without_not = p[1]
        karel = self.karel
        def fn():
            return getattr(karel, cond_without_not)()
        fn.tree = {
            'type': cond_without_not,
            'span': (p.lexpos(1), p.lexpos(1) + 1)
        }

        p[0] = fn

    def p_action(self, p):
        '''action : MOVE
                  | TURN_RIGHT
                  | TURN_LEFT
                  | PICK_MARKER
                  | PUT_MARKER
        '''
        action = p[1]
        karel = self.karel
        def fn():
            return getattr(karel, action)()
        fn.tree = {'type': action, 'span': (p.lexpos(1), p.lexpos(1) + 1)}
        p[0] = fn

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        def fn():
            return int(value)
        fn.tree = {'count': value, 'span': (p.lexpos(1), p.lexpos(1) + 1)}
        p[0] = fn

    def p_error(self, p):
        if p:
            raise KarelSyntaxError("Syntax error at '%s'" % p.value)
        else:
            raise KarelSyntaxError("Syntax error at EOF")


if __name__ == '__main__':
    parser = KarelForSynthesisParser()
    parser_prompt(parser)
