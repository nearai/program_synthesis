import struct

def read_index(filename):
    index = []
    with open(filename, 'rb') as index_file:
        while True:
            offset = index_file.read(8)
            if not offset:
                break
            offset, = struct.unpack('<Q', offset)
            index.append(offset)
    return index
