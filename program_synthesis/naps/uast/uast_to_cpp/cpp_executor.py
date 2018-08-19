from contextlib import contextmanager
import os
import subprocess
import shutil
import sys
import tempfile

from program_synthesis.naps.uast.uast_to_cpp.program_converter import program_to_cpp
from program_synthesis.naps.uast.uast_to_cpp.tests_converter import test_to_cpp


BASE_PATH = os.path.abspath(os.path.dirname(__file__))


class ProgramSourceGenerationError(Exception):
    pass


class ProgramCompilationError(Exception):
    pass


@contextmanager
def get_tempdir(cleanup):
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    if cleanup:
        shutil.rmtree(tmpdir)


def timeout_cmd(test_timeout):
    if "linux" in sys.platform:
        return ['timeout', str(test_timeout)]
    elif "darwin" == sys.platform:
        return ["gtimeout", str(test_timeout)]


def compile_run_program_and_tests(code_tree, tests, debug_info=False, cleanup=True, test_timeout=60):
    # Try to compile the program.
    passing_tests, test_compilation_errors, test_runtime_errors = [0]*3
    try:
        program_cpp, program_h = program_to_cpp(code_tree)
    except Exception as e:
        raise ProgramSourceGenerationError(e)

    with get_tempdir(cleanup) as tmpdir:
        lib_file = "lib_program.so"
        program_cpp_filepath = os.path.join(tmpdir, 'lib_program.cpp')
        with open(program_cpp_filepath, "w") as f:
            f.write(program_cpp)
        program_h_filepath = os.path.join(tmpdir, 'lib_program.h')
        with open(program_h_filepath, "w") as f:
            f.write(program_h)
        if "linux" in sys.platform:
            program_comp_res = subprocess.run(args=["clang++",
                                                    "-v",  # Verbose compilation.
                                                    "-pipe",  # Speed-up.
                                                    "-fPIC",
                                                    "-std=c++14", "-stdlib=libc++",
                                                    '-I', BASE_PATH,  # For special_lib.h
                                                    '-shared', '-undefined', 'dynamic_lookup',  # It's a shared lib.
                                                    '-o', lib_file, program_cpp_filepath], cwd=tmpdir,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:  #  "darwin" == sys.platform:
            program_comp_res = subprocess.run(args=["clang++",
                                                    "-v",  # Verbose compilation.
                                                    "-pipe",  # Speed-up.
                                                    "-std=c++14", "-stdlib=libc++",
                                                    '-I', BASE_PATH,  # For special_lib.h
                                                    '-shared', '-undefined', 'dynamic_lookup',  # It's a shared lib.
                                                    '-o', lib_file, program_cpp_filepath], cwd=tmpdir,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if program_comp_res.returncode != 0:
            if debug_info:
                print(program_comp_res.stderr.decode("utf-8"))
            raise ProgramCompilationError(program_comp_res.stderr.decode("utf-8"))

        # Try to compile and run the tests.
        for test_idx, test in enumerate(tests):
            test_file = "test%s" % test_idx
            test_cpp_filepath = os.path.join(tmpdir, 'test%s.cpp' % test_idx)
            try:
                with open(test_cpp_filepath, "w") as f:
                    f.write(test_to_cpp(code_tree, program_h, test))
            except:
                if debug_info:
                    print(program_cpp_filepath + ":1")
                    print(program_h_filepath + ":1")
                    print(test_cpp_filepath + ":1")
                test_compilation_errors += 1
                continue
            test_comp_res = subprocess.run(args=["clang++",
                                                 "-Wall",
                                                 "-v",  # Verbose compilation.
                                                 "-pipe",  # Speed-up.
                                                 "-std=c++14", "-stdlib=libc++",
                                                 '-I', BASE_PATH,  # For output_comparator.h.
                                                 '-I', tmpdir,  # For lib_program.h.
                                                 '-L', tmpdir,  # For lib_program.so.
                                                 '-l_program',
                                                 '-o', test_file, test_cpp_filepath], cwd=tmpdir,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if test_comp_res.returncode != 0:
                if debug_info:
                    print(test_comp_res.stderr.decode("utf-8"))
                    print(program_cpp_filepath + ":1")
                    print(program_h_filepath + ":1")
                    print(test_cpp_filepath + ":1")
                test_compilation_errors += 1
                continue

            subprocess.run(args=["chmod", "+x", test_file], cwd=tmpdir)
            env = os.environ
            if "linux" in sys.platform:
                if 'LD_LIBRARY_PATH' in env:
                    env['LD_LIBRARY_PATH'] = tmpdir + ':' + os.environ['LD_LIBRARY_PATH']
                else:
                    env['LD_LIBRARY_PATH'] = tmpdir
            test_run_res = subprocess.run(args=["./%s" % test_file], cwd=tmpdir,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            if test_run_res.returncode != 0:
                if debug_info:
                    print("%s\n%s" % (test_run_res.returncode, test_run_res.stderr.decode("utf-8")))
                test_runtime_errors += 1
                continue
            if test_run_res.stdout[:9] == b'INCORRECT':
                if debug_info:
                    print(test_run_res.stderr.decode("utf-8"))
                    print(test_run_res.stdout.decode("utf-8"))
                    print(program_cpp_filepath+":1")
                    print(program_h_filepath+":1")
                    print(test_cpp_filepath+":1")
            passing_tests += 1
    return passing_tests, test_compilation_errors, test_runtime_errors
