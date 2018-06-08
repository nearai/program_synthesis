import unittest

from program_synthesis.naps.uast import uast_pprint
from program_synthesis.naps.uast import uast_to_java


class UASTToJava(unittest.TestCase):

    def test_conversion(self):
        uast = {"funcs": [["func", "void", "__globals__.__init__", [], [], [["assign", "int", ["field", "int", ["var", "__globals__#", "__globals__"], "ParenthesisRemoval.mod"], ["invoke", "int", "+", [["invoke", "int", "*", [["invoke", "int", "*", [["val", "int", 1000], ["val", "int", 1000]]], ["val", "int", 1000]]], ["val", "int", 7]]]]]], ["ctor", "ParenthesisRemoval#", "ParenthesisRemoval.__init__", [], [], [["return", "void", ["var", "ParenthesisRemoval#", "this"]]]], ["func", "int", "ParenthesisRemoval.countWays", [["var", "ParenthesisRemoval#", "this"], ["var", "char*", "s"]], [["var", "int", "res"], ["var", "int", "cnt"], ["var", "int", "i"]], [["assign", "int", ["var", "int", "res"], ["val", "int", 1]], ["assign", "int", ["var", "int", "cnt"], ["val", "int", 0]], ["assign", "int", ["var", "int", "i"], ["val", "int", 0]], ["while", "void", ["invoke", "bool", "<", [["var", "int", "i"], ["invoke", "int", "len", [["var", "char*", "s"]]]]], [["if", "void", ["invoke", "bool", "==", [["invoke", "char", "array_index", [["var", "char*", "s"], ["var", "int", "i"]]], ["val", "char", 40]]], [["assign", "int", ["var", "int", "cnt"], ["invoke", "int", "+", [["var", "int", "cnt"], ["val", "int", 1]]]], ["var", "int", "cnt"]], [["assign", "int", ["var", "int", "res"], ["invoke", "int", "*", [["var", "int", "res"], ["cast", "int", ["var", "int", "cnt"]]]]], ["assign", "int", ["var", "int", "res"], ["invoke", "int", "%", [["var", "int", "res"], ["field", "int", ["var", "__globals__#", "__globals__"], "ParenthesisRemoval.mod"]]]], ["assign", "int", ["var", "int", "cnt"], ["invoke", "int", "-", [["var", "int", "cnt"], ["val", "int", 1]]]], ["var", "int", "cnt"]]]], [["assign", "int", ["var", "int", "i"], ["invoke", "int", "+", [["var", "int", "i"], ["val", "int", 1]]]], ["var", "int", "i"]]], ["return", "void", ["cast", "int", ["invoke", "int", "%", [["var", "int", "res"], ["field", "int", ["var", "__globals__#", "__globals__"], "ParenthesisRemoval.mod"]]]]]]], ["func", "int", "__main__", [["var", "char*", "s"]], [], [["return", "void", ["invoke", "int", "ParenthesisRemoval.countWays", [["invoke", "ParenthesisRemoval#", "ParenthesisRemoval.__init__", []], ["var", "char*", "s"]]]]]]], "types": [["record", "__globals__", {"ParenthesisRemoval.mod": ["var", "int", "mod"]}], ["record", "ParenthesisRemoval", {}]]}
        java = uast_to_java.uast_to_java(uast)
        self.assertEqual(java, """import java.util.*;
class ParenthesisRemoval {
static int mod = 1000 * 1000 * 1000 + 7;

public static int countWays(ParenthesisRemoval this_, String s) {
  int res;
  int cnt;
  int i;
  res = 1;
  cnt = 0;
  i = 0;
  while (i < s.length()) {
    if (s.charAt(i) == 40) {
      cnt = cnt + 1;
    }
    else {
      res = res * ((int)cnt);
      res = res % mod;
      cnt = cnt - 1;
    }
    i = i + 1;
  }
  return ((int)res % mod);
}

public static int __main__(String s) {
  return ParenthesisRemoval.countWays(new ParenthesisRemoval(), s);
}
}
""")


if __name__ == "__main__":
    unittest.main()
