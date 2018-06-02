import argparse
import os
import re
import struct

import tqdm

from program_synthesis.tools import indexed_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    piece_ids = []
    num_pieces = []

    for filename in args.input:
        m = re.search('(\d+)-of-(\d+)', filename)
        if not m:
            raise ValueError('{} lacks shard information')
        piece_ids.append(int(m.group(1)))
        num_pieces.append(int(m.group(2)))

    assert all(num_pieces[0] == p for p in num_pieces)
    num_pieces = num_pieces[0]
    piece_ids, filenames = zip(*sorted(zip(piece_ids, args.input)))
    assert piece_ids == tuple(range(num_pieces))

    output_file = open(args.output, 'w')
    index_file = open(args.output + '.index', 'w')

    index_offset = 0
    for filename in tqdm.tqdm(filenames):
        for v in indexed_file.read_index(filename + '.index'):
            index_file.write(struct.pack('<Q', v + index_offset))

        with open(filename) as piece_file:
            while True:
                chunk = piece_file.read(1024768)
                if not chunk:
                    break
                output_file.write(chunk)

        index_offset = output_file.tell()
