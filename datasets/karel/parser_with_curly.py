from __future__ import print_function

from .parser_base import dummy, get_hash, parser_prompt, Parser
from .utils import KarelSyntaxError


class KarelWithCurlyParser(Parser):
    """
    Parser for Karel programming language with curly braces.
    """

    tokens = [
            'DEF', 'RUN', 
            'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'SEMI', 'INT', #'NEWLINE',
            'WHILE', 'REPEAT',
            'IF', 'IFELSE', 'ELSE',
            'FRONT_IS_CLEAR', 'LEFT_IS_CLEAR', 'RIGHT_IS_CLEAR',
            'MARKERS_PRESENT', 'NO_MARKERS_PRESENT', 'NOT',
            'MOVE', 'TURN_RIGHT', 'TURN_LEFT',
            'PICK_MARKER', 'PUT_MARKER',
    ]

    t_ignore =' \t\n'
    t_LPAREN = '\('
    t_RPAREN = '\)'
    t_LBRACE = '\{'
    t_RBRACE = '\}'
    t_SEMI   = ';'

    t_DEF = 'def'
    t_RUN = 'run'
    t_WHILE = 'while'
    t_REPEAT = 'repeat'
    t_IF = 'if'
    t_IFELSE = 'ifelse'
    t_ELSE = 'else'
    t_NOT = 'not'

    t_FRONT_IS_CLEAR = 'front_is_clear'
    t_LEFT_IS_CLEAR = 'left_is_clear'
    t_RIGHT_IS_CLEAR = 'right_is_clear'
    t_MARKERS_PRESENT = 'markers_present'
    t_NO_MARKERS_PRESENT = 'no_markers_present'

    conditional_functions = [
            t_FRONT_IS_CLEAR, t_LEFT_IS_CLEAR, t_RIGHT_IS_CLEAR,
            t_MARKERS_PRESENT, t_NO_MARKERS_PRESENT,
    ]

    t_MOVE = 'move'
    t_TURN_RIGHT = 'turn_right'
    t_TURN_LEFT = 'turn_left'
    t_PICK_MARKER = 'pick_marker'
    t_PUT_MARKER = 'put_marker'

    action_functions = [
            t_MOVE,
            t_TURN_RIGHT, t_TURN_LEFT,
            t_PICK_MARKER, t_PUT_MARKER,
    ]

    #########
    # lexer
    #########

    def t_INT(self, t):
        r'\d+'

        value = int(t.value)
        if not (self.min_int <= value <= self.max_int):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`". \
                    format(self.min_int, self.max_int, value))

        t.value = value
        return t

    def random_INT(self):
        return self.rng.randint(self.min_int, self.max_int + 1)

    def t_error(self, t):
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)

    #########
    # parser
    #########

    def p_prog(self, p):
        '''prog : DEF RUN LPAREN RPAREN LBRACE stmt RBRACE'''
        stmt = p[6]
        p[0] = lambda: stmt()

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        fn = p[1]
        p[0] = lambda: fn()

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt SEMI stmt
        '''
        stmt1, stmt2 = p[1], p[3]
        def fn():
            stmt1(); stmt2();
        p[0] = fn

    def p_if(self, p):
        '''if : IF LPAREN cond RPAREN LBRACE stmt RBRACE
        '''
        cond, stmt = p[3], p[6]

        hit_info = self.hit_info
        if hit_info is not None:
            num = get_hash()
            hit_info[num] += 1

            def fn():
                if cond():
                    hit_info[num] -= 1
                    out = stmt()
                else:
                    out = dummy()
                return out
        else:
            fn = lambda: stmt() if cond() else dummy()

        p[0] = fn

    def p_ifelse(self, p):
        '''ifelse : IFELSE LPAREN cond RPAREN LBRACE stmt RBRACE ELSE LBRACE stmt RBRACE
        '''
        cond, stmt1, stmt2 = p[3], p[6], p[10]

        hit_info = self.hit_info
        if hit_info is not None:
            num1, num2 = get_hash(), get_hash()
            hit_info[num1] += 1
            hit_info[num2] += 1

            def fn():
                if cond():
                    hit_info[num1] -= 1
                    out = stmt1()
                else:
                    hit_info[num2] -= 1
                    out = stmt2()
                return out
        else:
            fn = lambda: stmt1() if cond() else stmt2()

        p[0] = fn

    def p_while(self, p):
        '''while : WHILE LPAREN cond RPAREN LBRACE stmt RBRACE
        '''
        cond, stmt = p[3], p[6]

        hit_info = self.hit_info
        if hit_info is not None:
            num = get_hash()
            hit_info[num] += 1

            def fn():
                while(cond()):
                    hit_info[num] -= 1
                    stmt()
        else:
            def fn():
                while(cond()):
                    stmt()
        p[0] = fn

    def p_repeat(self, p):
        '''repeat : REPEAT LPAREN cste RPAREN LBRACE stmt RBRACE
        '''
        cste, stmt = p[3], p[6]

        hit_info = self.hit_info
        if hit_info is not None:
            num = get_hash()
            hit_info[num] += 1

            def fn():
                for _ in range(cste()):
                    hit_info[num] -= 1
                    stmt()
        else:
            def fn():
                for _ in range(cste()):
                    stmt()
        p[0] = fn

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT cond_without_not
        '''
        if callable(p[1]):
            cond_without_not = p[1]
            fn = lambda: cond_without_not()
            p[0] = fn
        else: # NOT
            cond_without_not = p[2]
            fn = lambda: not cond_without_not()
            p[0] = fn

    def p_cond_without_not(self, p):
        '''cond_without_not : FRONT_IS_CLEAR LPAREN RPAREN
                            | LEFT_IS_CLEAR LPAREN RPAREN
                            | RIGHT_IS_CLEAR LPAREN RPAREN
                            | MARKERS_PRESENT LPAREN RPAREN 
                            | NO_MARKERS_PRESENT LPAREN RPAREN
        '''
        cond_without_not = p[1]
        p[0] = lambda: getattr(self.karel, cond_without_not)()

    def p_action(self, p):
        '''action : MOVE LPAREN RPAREN
                  | TURN_RIGHT LPAREN RPAREN
                  | TURN_LEFT LPAREN RPAREN
                  | PICK_MARKER LPAREN RPAREN
                  | PUT_MARKER LPAREN RPAREN
        '''
        action = p[1]
        def fn():
            return getattr(self.karel, action)()
        p[0] = fn

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        p[0] = lambda: int(value)

    def p_error(self, p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")
        raise KarelSyntaxError


if __name__ == '__main__':
    parser = KarelWithCurlyParser()
    parser_prompt(parser)
