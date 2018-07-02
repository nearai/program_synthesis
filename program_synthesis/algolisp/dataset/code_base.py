import six
import copy
import math


class MetaNode(type):
    node_types = {}
    attr_types = set()

    def __new__(mcls, name, bases, dct):
        attrs = list(dct.get('attrs', {}))
        for attr, _, _ in attrs:
            mcls.attr_types.add(attr)

        dct['attrs'] = []
        for base in bases:
            if hasattr(base, 'attrs'):
                dct['attrs'] += base.attrs
        dct['attrs'] += attrs

        new_class = type.__new__(mcls, name, bases, dct)
        mcls.node_types[name] = new_class
        return new_class


@six.add_metaclass(MetaNode)
class Node(object):
    attrs = ()

    def __init__(self, *args, **kwargs):
        self.set_attrs(*args, **kwargs)

    def set_attrs(self, *args, **kwargs):
        values = kwargs

        cargs = 0
        for (attr_name, attr_type, _), value in zip(self.attrs, args):
            if attr_name in values:
                raise ValueError("Unexpected positional argument: %s, expected: %s" % (value, attr_name))
            if attr_type is not None:
                if isinstance(attr_type, tuple):
                    attr_type, var_type = attr_type
                    if not isinstance(value.get_type(), var_type):
                        raise ValueError("Unexpected type of var/expr %s.%s argument: %s, expected: %s" % (
                            type(self).__name__, attr_name, type(value.get_type()), var_type
                        ))
                if not isinstance(value, attr_type):
                    raise ValueError("Unexpected type of %s.%s argument: %s, expected: %s" % (
                        type(self).__name__, attr_name, type(value), attr_type
                    ))
            setattr(self, attr_name, value)
            cargs += 1

        for attr_name, attr_type, attr_default in self.attrs[len(args):]:
            if attr_name in values:
                value = values[attr_name]
                cargs += 1
            else:
                value = copy.deepcopy(attr_default)
            if attr_type is not None:
                if isinstance(attr_type, tuple):
                    attr_type, var_type = attr_type
                    if not isinstance(value.get_type(), var_type):
                        raise ValueError("Unexpected type of var/expr %s.%s argument: %s, expected: %s" % (
                            type(self).__name__, attr_name, type(value), attr_type
                        ))
                if not isinstance(value, attr_type):
                    raise ValueError("Unexpected type of %s.%s argument: %s, expected: %s" % (
                        type(self).__name__, attr_name, type(value), attr_type
                    ))
            setattr(self, attr_name, value)

        if cargs != len(args) + len(values):
            raise ValueError("Unexpected arguments: %s" % (values))

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        for attr, _, _ in self.attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        attrs = ['%s=%s' % (attr, getattr(self, attr))
                 for attr, _, _ in self.attrs]
        if attrs:
            return "%s[%s]" % (
                type(self).__name__,
                ','.join(attrs))
        return type(self).__name__
