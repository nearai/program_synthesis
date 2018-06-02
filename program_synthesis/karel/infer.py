import argparse
import collections
from six.moves import cPickle as pickle
import struct

import torch
import tqdm

from program_synthesis.karel import arguments
from program_synthesis.karel import dataset
from program_synthesis.karel import models
from program_synthesis import tools


def infer(args):
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    tools.restore_args(args)
    arguments.backport_default_args(args)
    dataset.set_vocab(args)
    m = models.get_model(args)
    if m.last_step == 0:
        raise ValueError('Attempting to evaluate on untrained model')

    if args.eval_final:
        eval_dataset = dataset.get_eval_final_dataset(args, m)
    elif args.eval_train:
        eval_dataset = dataset.get_train_dataset(args, m, for_eval=True)
    else:
        eval_dataset = dataset.get_eval_dataset(args, m)
    m.model.eval()

    f = open(args.infer_output, 'w')
    index_f = open(args.infer_output + '.index', 'w')
    infer_counters = collections.Counter()
    num_outputs = 0
    iterator = tqdm.tqdm(eval_dataset, dynamic_ncols=True)
    for batch in iterator:
        infer_results = m.inference(batch)
        infer_outputs = m.process_infer_results(batch, infer_results,
                infer_counters)
        for output in infer_outputs:
            offset = f.tell()
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
            index_f.write(struct.pack('<Q', offset))
            num_outputs += 1
            if args.infer_limit and num_outputs >= args.infer_limit:
                return

        iterator.set_postfix(**infer_counters)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')

    args = parser.parse_args()
    assert args.infer_output
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.model_type or (not args.model_dir and args.model_type != 'search'):
        raise ValueError("Specify model_dir and model_type")
    infer(args)
