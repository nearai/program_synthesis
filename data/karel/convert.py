import argparse
import json
import cPickle as pickle
import struct

import numpy as np
import tqdm

def str_to_arr(s):
    result = [int(v.split(':')[0]) for v in s.split(' ')]
    result.sort()
    return np.array(result, dtype=np.uint16)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    f = open(args.output, 'w')
    index_f = open(args.output + '.index', 'w')
    for i, line in tqdm.tqdm(enumerate(open(args.input))):
        obj = json.loads(line)
        offset = f.tell()
        pickle.dump({
            'id': i,
            'guid': obj['guid'],
            'examples': [{
                'in': str_to_arr(ex['inpgrid_tensor']),
                'out': str_to_arr(ex['outgrid_tensor']),
            } for ex in obj['examples']],
            'code': obj['program_tokens'],
        }, f, pickle.HIGHEST_PROTOCOL)
        index_f.write(struct.pack('<Q', offset))
