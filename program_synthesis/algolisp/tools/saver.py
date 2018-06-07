"""Tools to save/restore model from checkpoints."""

import argparse
import sys
import os
import torch
import re
import json

CHECKPOINT_PATTERN = re.compile('^checkpoint-(\d+)$')


class ArgsDict(dict):

    def __init__(self, **kwargs):
        super(ArgsDict, self).__init__()
        for key, value in kwargs.items():
            self[key] = value
        self.__dict__ = self


def load_checkpoint(model, optimizer, model_dir, map_to_cpu=False, step=None):
    path = os.path.join(model_dir, 'checkpoint')
    if step is not None:
        path += '-{:08d}'.format(step)
    if os.path.exists(path):
        print("Loading model from %s" % path)
        if map_to_cpu:
            checkpoint = torch.load(
                path, map_location=lambda storage, location: storage)
        else:
            checkpoint = torch.load(path)
        old_state_dict = model.state_dict()
        for key in old_state_dict.keys():
            if key not in checkpoint['model']:
                checkpoint['model'][key] = old_state_dict[key]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint.get('step', 0)
    return 0


def load_and_map_checkpoint(model, model_dir, remap):
    path = os.path.join(model_dir, 'checkpoint')
    print("Loading parameters %s from %s" % (remap.keys(), model_dir))
    checkpoint = torch.load(path)
    new_state_dict = model.state_dict()
    for name, value in remap.items():
        # TODO: smarter mapping.
        new_state_dict[name] = checkpoint['model'][value]
    model.load_state_dict(new_state_dict)


def save_checkpoint(model, optimizer, step, model_dir, ignore=[],
                    keep_every_n=10000000):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    path = os.path.join(model_dir, 'checkpoint')
    step_padded = format(step, '08d')
    state_dict = model.state_dict()
    if ignore:
        for key in state_dict.keys():
            for item in ignore:
                if key.startswith(item):
                    state_dict.pop(key)
    torch.save({
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'step': step
    }, '{}-{}'.format(path, step_padded))
    if os.path.exists(path):
        os.unlink(path)
    source = 'checkpoint-' + step_padded
    os.symlink(source,  path)

    # Cull old checkpoints.
    if keep_every_n is not None:
        all_checkpoints = []
        for name in os.listdir(model_dir):
            m = CHECKPOINT_PATTERN.match(name)
            if m is None or name == source:
                continue
            checkpoint_step = int(m.group(1))
            all_checkpoints.append((checkpoint_step, name))
        all_checkpoints.sort()

        last_step = float('-inf')
        for checkpoint_step, name in all_checkpoints:
            if checkpoint_step - last_step >= keep_every_n:
                last_step = checkpoint_step
                continue
            os.unlink(os.path.join(model_dir, name))


class Saver(object):
    """Class to manage save and restore for the model and optimizer."""

    def __init__(self, model, optimizer, keep_every_n=None):
        self._model = model
        self._optimizer = optimizer
        self._keep_every_n = keep_every_n

    def restore(self, model_dir, map_to_cpu=False, step=None):
        """Restores model and optimizer from given directory.

        Returns:
           Last training step for the model restored.
        """
        last_step = load_checkpoint(
            self._model, self._optimizer, model_dir, map_to_cpu, step)
        return last_step

    def save(self, model_dir, step):
        """Saves model and optimizer to given directory.

        Args:
           model_dir: Model directory to save.
           step: Current training step.
        """
        save_checkpoint(self._model, self._optimizer, step, model_dir,
                        keep_every_n=self._keep_every_n)

    def restore_part(self, other_model_dir, remap):
        """Restores part of the model from other directory.

        Useful to initialize part of the model with another pretrained model.

        Args:
            other_model_dir: Model directory to load from.
            remap: dict, remapping current parameters to the other model's.
        """
        load_and_map_checkpoint(self._model, other_model_dir, remap)


def save_args(args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(os.path.join(args.model_dir, 'args.json'), 'w') as f:
        f.write(json.dumps(vars(args)))


def restore_args(args):
    if not os.path.exists(args.model_dir):
        raise Exception('{} does not exist'.format(args.model_dir))
    with open(os.path.join(args.model_dir, 'args.json')) as f:
        new_args = json.loads(f.read())
    for arg in new_args:
        if not hasattr(args, arg):
            setattr(args, arg, new_args[arg])


def print_params(dct, indent=0):
    for key in dct:
        if isinstance(dct[key], dict):
            print(" " * indent + str(key))
            print_params(dct[key], indent + 2)
        elif isinstance(dct[key], torch.Tensor):
            print(" " * indent + key + " " + str(dct[key].size()))
        else:
            print(" " * indent + key + " = " + str(dct[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checkpoint Viewer')
    parser.add_argument('--model_dir', type=str, default='')
    args = parser.parse_args()

    path = os.path.join(args.model_dir, 'checkpoint')
    print("Loading model from %s" % path)
    checkpoint = torch.load(path)
    print_params(checkpoint)

