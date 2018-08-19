from enum import Enum


class CPPLibs(Enum):
    string = 1
    vector = 2
    set = 3
    map = 4
    memory = 5
    cmath = 6
    locale = 7
    algorithm = 8
    cstdlib = 9
    special = 10  # Our own lib.


def convert_libs(libs):
    if not libs:
        return ""
    if CPPLibs.special in libs:
        return """
        #include <special_lib.h>
        using namespace std;
        """
    result = []
    for lib in libs:
        result.append("#include<%s>" % str(lib).split('.')[1])
    result.append('using namespace std;')
    return "\n".join(result)
