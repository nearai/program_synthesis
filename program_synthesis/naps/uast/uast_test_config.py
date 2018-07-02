import six
import numbers

print_dump = False # print the dump of the Java or C++ tree when a test fails
print_ret = True   # print the dump of UAST when a test fails


def assert_value(a, b):
    assert test_passed(a, b), "%s != %s" % (a, b)


def test_passed(a, b):
    if isinstance(a, six.string_types):
        a = a.strip()
    if isinstance(b, six.string_types):
        b = b.strip()
    if ((isinstance(a, float) and isinstance(b, numbers.Number)) or
        (isinstance(b, float) and isinstance(a, numbers.Number))):
        return abs(a - b) < 1e-6 or (a != 0 and b != 0 and abs((a - b) / min(abs(a), abs(b))) < 1e-6)
    return a == b
