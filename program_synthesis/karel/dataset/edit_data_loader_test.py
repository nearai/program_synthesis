import os
import sys

import mock
import pytest

from program_synthesis.karel import arguments
from program_synthesis.karel.dataset import dataset
from program_synthesis.karel.dataset import edit_data_loader
from program_synthesis.karel.models import karel_edit_model


@pytest.fixture
def args():
    with mock.patch('sys.argv', [
            'test', '--num_placeholders', '0', '--karel-merge-io', 'setlstm'
    ]):
        args = arguments.parse('', 'train')
        args.word_vocab = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                '../../../data/karel/word.vocab'))
        return args


def test_get_batch_sync(args):
    karel_dataset = dataset.KarelTorchDataset(
            dataset.relpath('../../data/karel/train.pkl'))
    batch_size = 32
    m = karel_edit_model.KarelStepEditModel(args)
    batch_processor = m.batch_processor(for_eval=False)

    loader = edit_data_loader.SynchronousKarelEditDataLoader(
            karel_dataset,
            batch_size,
            batch_processor)
    batch = next(iter(loader))


def test_exhaustion_sync(args):
    karel_dataset = dataset.KarelTorchDataset(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'testdata',
                'test-cs106a-line1.pkl'))
    batch_size = 2
    m = karel_edit_model.KarelStepEditModel(args)
    batch_processor = m.batch_processor(for_eval=True)

    loader = edit_data_loader.SynchronousKarelEditDataLoader(
            karel_dataset,
            batch_size,
            batch_processor,
            beam_size=None,
            shuffle=False)

    all_batches = list(loader)
    assert sum(len(b.orig_examples) for b in all_batches) == 16

    # Run again
    all_batches2 = list(loader)
    assert sum(len(b.orig_examples) for b in all_batches2) == 16

    # Test that all of the outputs are unique
    all_codes = [e.cur_code for b in all_batches for e in b.orig_examples]
    assert len(set(all_codes)) == len(all_codes)
