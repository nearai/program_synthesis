"""
Implements base class for the pipes. Interface is compatible with Python file objects.
"""


class Pipe(object):
    """
    A base class for all pipes. Specific pipe implementation is expected to override the following functions:
    __init__ : use it to record the pipe settings, e.g. the filename;
    __iter__: use it to iterate over the content of the pipe, expects __enter__ to be called before that;
    enter: here we allocate the resources, e.g. open the file or the connection to the database;
    exit: here we deallocate the resources;
    __get_item__ and __len__: implement it if the pipeline supports random access.

    The class object might have the following attribute:
    input (Pipe object) that's what we input into the pipe.
    """

    def __iter__(self):
        raise NotImplementedError("The pipe %s does not have an output." % self.__class__.__name__)

    def enter(self):
        # By default the pipeline has its own resources to allocate.
        pass

    def exit(self):
        # By default the pipeline has its own resources to deallocate.
        pass

    def __enter__(self):
        if hasattr(self, 'input') and self.input is not None:
            self.input.__enter__()
        self.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()
        if hasattr(self, 'input') and self.input is not None:
            self.input.__exit__(exc_type, exc_val, exc_tb)

    def __getitem__(self, item):
        raise NotImplementedError("The pipe %s does not support random access." % self.__class__.__name__)

    def __len__(self):
        raise NotImplementedError("The length of the content of the pipe %s is not accessible."
                                  % self.__class__.__name__)


class CallablePipe(Pipe):
    """
    Wrapper to simplify adding a simple pipe that can be represented with a callable.
    """
    def __init__(self, callable_):
        self._callable = callable_

    def __iter__(self):
        return (self._callable(el) for el in self.input)

    def __getitem__(self, item):
        return self._callable(self.input[item])

    def __len__(self):
        return len(self.input)
