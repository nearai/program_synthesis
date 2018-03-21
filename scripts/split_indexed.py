import argparse
import os
import struct

import tqdm

from program_synthesis.datasets import indexed_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--num-pieces', required=True, type=int)
    args = parser.parse_args()

    root, ext = os.path.splitext(args.input)

    index = []
    with open(args.input + '.index') as index_file:
        while True:
            offset = index_file.read(8)
            if not offset:
                break
            offset, = struct.unpack('<Q', offset)
            index.append(offset)

    num_elems = len(index)
    pieces_num_elems = [num_elems // args.num_pieces] * args.num_pieces
    pieces_num_elems[0] += num_elems - sum(pieces_num_elems)
    index.append(os.stat(args.input).st_size)

    input_file = open(args.input)

    index_offset = 0
    for i, piece_num_elems in tqdm.tqdm(enumerate(pieces_num_elems)):
        piece_name  = '{}-{:03d}-of-{:03d}{}'.format(
                root, i, args.num_pieces, ext)

        piece_start = index[index_offset]
        piece_end = index[index_offset + piece_num_elems]
        piece_size = piece_end - piece_start
        input_file.seek(piece_start)
        with open(piece_name, 'w') as output_file:
            total_written = 0
            while total_written < piece_size:
                chunk = input_file.read(
                        min(1024768, piece_size - total_written))
                assert chunk, 'EOF reached unexpectedly'
                output_file.write(chunk)
                total_written += len(chunk)

        piece_index = [
            v - piece_start
            for v in index[index_offset:index_offset + piece_num_elems]
        ]
        with open(piece_name + '.index', 'w') as index_file:
            for v in piece_index:
                index_file.write(struct.pack('<Q', v))

        index_offset += piece_num_elems
