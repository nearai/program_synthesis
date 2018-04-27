class OutputStream(object):
    def open(self):
        raise NotImplementedError

    def write(self, data):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def read(self, yield_none_if_waiting=False):
        raise NotImplementedError
