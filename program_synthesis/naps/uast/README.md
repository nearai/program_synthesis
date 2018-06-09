# UAST

Universal AST.

## Executing a UAST

    executor = Executor(uast)
    ret_val = executor.execute_func(func_name, inputs)

the ctor of Executor invokes a function called "__globals__.__init__" to populate an
instance of variable "__globals__" that is available to all the functions

Therefore, a function called "__globals__.__init__" should be present if any global variables exist.

generally the entry point for the uast is a function called "__main__" (but technically
`execute_func` can call any function, not necessarily "__main__")


## UAST Specification

Lower case (e.g. name) is terminal symbols
Upper case (e.g. VAR) is non-terminal symbols
Strings (e.g. 'record') is string constants



PROGRAM     ::= {'types': [RECORD...], 'funcs': [FUNC...]}

                - a dictionary containing a list of records (structs) and a list of functions
                - one record is mandatory: __globals__, which has all the global variables
                - two functions are mandatory:
                   - __globals__.__init__ initializes all the fields of __globals__
                   - __main__ is the entry point

RECORD      ::= ['record', name, {field_name: VAR...}]

FUNC        ::= ['func', TYPE, name, [VAR...], [VAR...], [STMT...]]
              | ['ctor', TYPE, name, [VAR...], [VAR...], [STMT...]]
              
                - the second entry (TYPE) is the return type
                - the fourth entry ([VAR...]) is the list of function arguments
                - the fifth entry ([VAR...]) is the list of function variables
                - the sixth entry ([STMT...]) is the function body
                - there's always an implicit variable called '__globals__' visible from all
                  functions
                - the difference between the 'ctor' and 'func' is that 'ctor' has an implicit
                  variable named 'this' that has the same type as the return type, and must
                  return it.
                - if 'func' is a result of converting an instance method from a higher level
                  language, 'this' would be explicitly present as the first argument.

VAR         ::= ['var', TYPE, name]

STMT        ::= EXPR | IF | FOREACH | WHILE | BREAK | CONTINUE | RETURN | NOOP

EXPR        ::= ASSIGN | VAR | FIELD | CONSTANT | INVOKE | TERNARY | CAST

ASSIGN      ::= ['assign', TYPE, LHS, EXPR]
                - computes the EXPR ans assigns it into the LHS

LHS         ::= VAR | FIELD | INVOKE*

                - the invoke is only suitable if the function it invokes is 'array_index'
                - if the LHS if FIELD, the record parameter of the FIELD must be an LHS
                - if the LHS is INVOKE of 'array_index', the first argument of array_index must
                  be an LHS

IF          ::= ['if', TYPE, EXPR, [STMT...], [STMT...]]
                - EXPR is condition, first [STMT...] is then, second is else
                - TYPE is always 'void'

FOREACH     ::= ['foreach', TYPE, VAR, EXPR, [STMT...]]

                - EXPR should evaluate to an array, list or set
                - assignts each element of EXPR to VAR and executes the [STMT...]
                - TYPE is always 'void'

WHILE       ::= ['while', TYPE, EXPR, [STMT...], [STMT...]]

                - two [STMT...] lists are the body and the increment blocks
                - executes the body blocks for as long as the EXPR evaluates to true
                - executes the increment block after every body block
                - the increment block is separate from the body block to properly handle
                  CONTINUE and BREAK
                - TYPE is always 'void'

BREAK       ::= ['break', TYPE]

                - breaks execution from the innermost WHILE or FOREACH
                - the TYPE is always 'void'

CONTINUE    ::= ['continue', TYPE]

                - continues to the next iteration of the innermost WHILE or FOREACH
                - the TYPE is always 'void'

RETURN      ::= ['return', TYPE, EXPR]

                - aborts the execution of the innermost function and returns the value of EXPR
                - the TYPE is always 'void', RETURN is a statement, not an expression

NOOP        ::= ['noop']

FIELD       ::= ['field', TYPE, EXPR, field_name]

                - EXPR must evaluate to an instance of a record
                - field_name is a string
                - returns the value of the field of the instance of the record
                - if EXPR is an LHS, FIELD can be used as an LHS

CONSTANT    ::= ['val', TYPE, value]

INVOKE      ::= ['invoke', TYPE, function_name, [EXPR...]]

                - function_name is either:
                    - one of the FUNCs in the PROGRAM
                    - one of the standard functions (see uast.py)
                - TYPE should be the return type of the function

TERNARY     ::= ['?:', TYPE, EXPR, EXPR, EXPR]

                - EXPR is condition, first EXPR is then, second is else

CAST        ::= ['cast', TYPE, EXPR]

                - casts given EXPR to the given TYPE.
                - The primary use case is casting to 'string'
                - When converting Java to UAST, all the java casts are converted to UAST casts
                - TODO: ultimately the casts should be removed from UAST

TYPE        ::= object | bool | char | int | real | TYPE* | TYPE% | <TYPE|TYPE> | record_name#

                - TYPE* is an array of TYPE
                - TYPE% is a set of TYPE
                - <TYPE|TYPE> is a map of TYPE
                - strings are char*
                - record_name# is the type for record with name record_name
                   - e.g. "Point#" is the type for instances of record Point
                - example: "<char*|int%>" is a map from string to set of ints

