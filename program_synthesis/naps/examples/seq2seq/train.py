from itertools import chain
import os
import tqdm

import torch
from torch import optim

from program_synthesis.common.tools.reporter import TensorBoardWriter
from program_synthesis.common.tools import saver as saver_lib

from program_synthesis.naps.pipelines import read_naps
from program_synthesis.naps.examples.seq2seq.batch_preparator import SPECIAL_SIZE
from program_synthesis.naps.examples.seq2seq.seq2seq_model import SequentialText2Uast
from program_synthesis.naps.examples.seq2seq import arguments


def load_vocab(args):
    vocab_path = os.path.join(args.model_dir, "naps.vocab.txt")

    if not os.path.isfile(vocab_path):
        print("Building vocab")
        trainA, trainB, test = read_naps.read_naps_dataset()
        vocab = dict()
        with trainA, trainB, test, tqdm.tqdm() as pbar:
            pipeline = chain(trainA, trainB, test)
            for d in pipeline:
                for token in d["text"]:
                    vocab[token] = vocab.get(token, 0) + 1
                for token in d["code_sequence"]:
                    vocab[token] = vocab.get(token, 0) + 1
                pbar.update(1)
        # Sort for determinism.
        vocab = sorted(k for k, v in vocab.items() if v >= args.vocab_min_occurrences)
        with open(vocab_path, "w") as f:
            f.write("\n".join(vocab))
    else:
        vocab = []
        with open(vocab_path) as f:
            for token in f:
                vocab.append(token.strip("\n"))
    return dict((t, i + SPECIAL_SIZE) for i, t in enumerate(vocab))


def launch_training(args):
    print("Training.")
    print("Model args:\n%s" % '\n'.join("%s:\t%s" % (k, vars(args)[k]) for k in sorted(vars(args).keys())))

    train_data, _ = read_naps.read_naps_dataset_batched(batch_size=args.batch_size,
                                                        trainB_weight=0.3,
                                                        shuffle_variables=args.shuffle_variables,
                                                        sort_batch=True)
    vocab = load_vocab(args)
    model = SequentialText2Uast(args, vocab)

    # Initialize special objects.
    saver_lib.save_args(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.optimizer_weight_decay)
    saver = saver_lib.Saver(model, optimizer, args.saver_keep_every_n)
    reporter = TensorBoardWriter(args.model_dir)

    # Initialize training variables.
    last_step = saver.restore(args.model_dir, map_to_cpu=args.restore_map_to_cpu, step=None)
    last_loss = None
    last_saved_step = 0
    model.train()

    with train_data, tqdm.tqdm(smoothing=0.1) as pbar:
        for batch in train_data:
            # Update learning rate of the optimizer.
            lr = args.lr * args.lr_decay_rate ** (last_step // args.lr_decay_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            curr_loss = float(loss.data[0])
            if last_loss and last_step > 1000 and curr_loss > last_loss * 3:
                pbar.write("Loss exploded: %f to %f. Skipping batch." % (last_loss, curr_loss))
                continue
            else:
                last_loss = curr_loss
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)

            optimizer.step()

            # Update variables, save and report.
            last_step += 1
            if (last_step - last_saved_step) > args.saver_save_every_n:
                last_saved_step = last_step
                saver.save(args.model_dir, last_step)
                pbar.write("Saved model to %s" % os.path.abspath(args.model_dir))

            metrics = {'loss': curr_loss, 'lr': lr}
            for key, value in metrics.items():
                reporter.add(last_step, key, value)
            pbar.update(1)
            pbar.write("loss: {loss:.6f};\tlr: {lr:.8f}".format(**metrics))
    reporter.close()


if __name__ == "__main__":
    args = arguments.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    launch_training(args)
