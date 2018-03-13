import argparse
import collections
from six.moves import cPickle as pickle
import struct

import torch
import tqdm

from program_synthesis import arguments
from program_synthesis import datasets
from program_synthesis import models
from program_synthesis import tools


def infer(args):
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    tools.restore_args(args)
    arguments.backport_default_args(args)
    datasets.set_vocab(args)
    m = models.get_model(args)

    if args.eval_final:
        eval_dataset = datasets.get_eval_final_dataset(args, m)
    elif args.eval_train:
        eval_dataset, _ = datasets.get_dataset(args, m)
    else:
        eval_dataset = datasets.get_eval_dataset(args, m)
    m.model.eval()

    f = open(args.infer_output, 'w')
    index_f = open(args.infer_output + '.index', 'w')
    infer_counters = collections.Counter()
    for batch in tqdm.tqdm(eval_dataset):
        infer_results = m.inference(batch)
        infer_outputs = m.process_infer_results(batch, infer_results,
                infer_counters)
        for output in infer_outputs:
            for example in output['examples'][:5]:
                assert len(example['trace'].grids) >= 2
            offset = f.tell()
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
            index_f.write(struct.pack('<Q', offset))
        print(infer_counters)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')

    args = parser.parse_args()
    assert args.infer_output
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.model_type or (not args.model_dir and args.model_type != 'search'):
        raise ValueError("Specify model_dir and model_type")
    infer(args)
