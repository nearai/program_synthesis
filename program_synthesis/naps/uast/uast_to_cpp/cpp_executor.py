import os
import subprocess
import tempfile

from program_synthesis.naps.uast.uast_to_cpp.program_converter import program_to_cpp


BASE_PATH = os.path.abspath(os.path.dirname(__file__))


def execute_program(code_tree, tests):
    cpp, needs_special = program_to_cpp(code_tree)
    tmpdir = tempfile.mkdtemp()
    lib_file = "program.so"
    source_file = os.path.join(tmpdir, 'program.cpp')
    with open(source_file, "w") as f:
        f.write(cpp)
    res = subprocess.run(args=["clang++", "-v", "-std=c++14", "-stdlib=libc++", '-I', BASE_PATH,
                               '-shared', '-undefined', 'dynamic_lookup',
                               '-o', lib_file, source_file], cwd=tmpdir, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    print(res.stderr.decode("utf-8"))
    pass
