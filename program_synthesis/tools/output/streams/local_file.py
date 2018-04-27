from program_synthesis.tools.output.streams.base import OutputStream


class LocalFileOutputStream(OutputStream):
    def __init__(self, file_path):
        self._file_path = file_path
        self._file = None

    def open(self):
        self._file = open(self._file_path, 'w')

    def write(self, data):
        if self._file is None:
            self.open()

        self._file.write(data)
        self._file.flush()

    def close(self):
        self._file.close()

    def read(self):
        if self._file is None:
            self.open()
        yield self._file.read()
