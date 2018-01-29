from __future__ import print_function

from .parser_base import dummy, get_hash, parser_prompt, Parser
from .utils import KarelSyntaxError


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

    #########
    # parser
    #########

    def p_prog(self, p):
        '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
        stmt = p[4]

        @self.callout
        def fn():
            return stmt()
        p[0] = fn

    def p_stmt(self, p):
        '''stmt : while
                | repeat
                | stmt_stmt
                | action
                | if
                | ifelse
        '''
        function = p[1]

        @self.callout
        def fn():
            return function()
        p[0] = fn

    def p_stmt_stmt(self, p):
        '''stmt_stmt : stmt stmt
        '''
        stmt1, stmt2 = p[1], p[2]

        @self.callout
        def fn():
            stmt1(); stmt2();
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
        p[0] = fn

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT C_LBRACE cond_without_not C_RBRACE
        '''
        if callable(p[1]):
            cond_without_not = p[1]
            fn = lambda: cond_without_not()
            p[0] = fn
        else: # NOT
            cond_without_not = p[3]
            fn = lambda: not cond_without_not()
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
        p[0] = fn

    def p_cste(self, p):
        '''cste : INT
        '''
        value = p[1]
        p[0] = lambda: int(value)

    def p_error(self, p):
        if p:
            raise KarelSyntaxError("Syntax error at '%s'" % p.value)
        else:
            raise KarelSyntaxError("Syntax error at EOF")


if __name__ == '__main__':
    parser = KarelForSynthesisParser()
    parser_prompt(parser)
