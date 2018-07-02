import six

from program_synthesis.algolisp.dataset.code_base import Node, MetaNode

# Types
class Type(Node):

    def compatible(self, other_type):
        assert not isinstance(other_type, (AnyFuncType, AnyType)), "Should not compare %s with %s" % (self, other_type)
        if self.__class__ == FuncType and other_type.__class__ == FuncType:
            try:
                return (self.return_type.compatible(other_type.return_type) and 
                    ((other_type.argument_types in [AnyArgs.T, AnyArgs.T2] and 
                        len(self.argument_types) >= other_type.argument_types.min_args) or 
                    (len(self.argument_types) == len(other_type.argument_types) and 
                        all([their.compatible(mine) 
                            for (mine, their) in 
                            zip(self.argument_types, other_type.argument_types)]))))
            except:
                print("Exception while comparing types: %s and %s" % (self, other_type))
                raise
        if self.__class__ == Array and other_type.__class__ == Array:
            return self.underlying_type is None or self.underlying_type.compatible(other_type.underlying_type)

        return issubclass(self.__class__, other_type.__class__)

    def __hash__(self):
        return hash(str(self.__class__))


class NotFuncType(Type):
    def compatible(self, other_type):
        if isinstance(other_type, FuncType):
            return False
        return super(NotFuncType, self).compatible(other_type)


Void = MetaNode('Void', (NotFuncType,), {})
Boolean = MetaNode('Boolean', (NotFuncType,), {})
Number = MetaNode('Number', (NotFuncType,), {})
String = MetaNode('String', (NotFuncType,), {})

Type.T = Type()
NotFuncType.T = NotFuncType()
Void.T = Void()
Boolean.T = Boolean()
Number.T = Number()
String.T = String()


class TypeType(Type):
    attrs = (("base", Type, Void.T), )

    def compatible(self, other):
        assert not isinstance(other, AnyTypeType), "Should not compare %s with AnyTypeType" % self
        if not issubclass(other.__class__, TypeType):
            return False
        return self.base.compatible(other.base)


class AnyTypeType(TypeType):

    def compatible(self, other):
        return issubclass(other.__class__, TypeType)


class FuncType(Type):
    attrs = (('return_type', Type, None), ('argument_types', list, []))


class AnyType(Type):
    def compatible(self, other):
        return True


class AnyNotFuncType(NotFuncType):
    def compatible(self, other):
        return not isinstance(other, FuncType)


class AnyArgs(list):
    def __init__(self, min_args):
        self.min_args = min_args


TypeType.T = TypeType(Type.T)
AnyTypeType.T = AnyTypeType()
AnyType.T = AnyType()
AnyArgs.T = AnyArgs(1)
AnyArgs.T2 = AnyArgs(2)
AnyArgs.T3 = AnyArgs(3)
AnyNotFuncType.T = AnyNotFuncType()


class AnyFuncType(FuncType):
    def __init__(self):
        super(AnyFuncType, self).__init__(Type.T, AnyArgs.T)

    def compatible(self, other):
        return isinstance(other, FuncType) and (other.argument_types in [AnyArgs.T, AnyArgs.T2] or len(other.argument_types) > 0)


AnyFuncType.T = AnyFuncType()


class Enumerable(NotFuncType):
    pass


class Array(Enumerable):
    attrs = (('underlying_type', Type, None),)

    def __hash__(self):
        return hash(self.underlying_type)

    def __str__(self):
        return str(self.underlying_type) + "[]"


class Dict(Enumerable):
    attrs = (('key_type', Type, None), ('value_type', Type, None))

    def __hash__(self):
        return hash(self.key_type) + hash(self.value_type)

    def __str__(self):
        return "{%s:%s}" % (str(self.key_type), str(self.value_type))


_TYPES = {
    'any': Type.T,
    'int': Number.T,
    'string': String.T,
    'int[]': Array(Number.T),
    'string[]': Array(String.T),
    'bool': Boolean.T,
    'int[][]': Array(Array(Number.T)),
    'dict<int,int>': Dict(Number.T, Number.T),
    'dict<string,int>': Dict(String.T, Number.T),
}
_REV_TYPES = {v: k for k, v in _TYPES.items()}


def str_to_type(type_str):
    return _TYPES[type_str]


def type_to_str(type_):
    return _REV_TYPES[type_]


def any_type(type_):
    return type_ == AnyType.T or type_ == AnyFuncType.T


class FuncSignature(object):

    def __init__(self, args, return_type, computed_type):
        self.args = args
        self.return_type = return_type
        self.computed_type = computed_type

    def compatible(self, func):
        pass

    def compatible_args(self, args):
        pass

    def partial(self, expected_return_type, some_args):
        pass
