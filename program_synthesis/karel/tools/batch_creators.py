import numpy as np


class ConstantBatch(object):
    def __init__(self, eg_shown, eg_hidden, is_shuffled):
        self.eg_shown = eg_shown
        self.eg_hidden = eg_hidden
        self.is_shuffled = is_shuffled
    def randomize(self):
        pass
    def from_examples(self, all_examples):
        train_eg = all_examples[:self.eg_shown]
        val_only_eg = all_examples[self.eg_shown:self.eg_shown + self.eg_hidden]
        return train_eg, val_only_eg
    def __repr__(self):
        return "ConstantBatch({}, {}, {})".format(self.eg_shown, self.eg_hidden, self.is_shuffled)


class RandomBatch(object):
    def __init__(self, is_shuffled, *shown_pdf):
        """
        shown_pdf : a list of probabilities [p_1, p_2, p_3, ... p_n], where p_i is the probability
            of picking i for the shown probability distribution.
        """
        self.is_shuffled = is_shuffled
        self.shown_pdf = shown_pdf
    def randomize(self):
        self.eg_shown = np.random.choice(list(range(1, 1 + len(self.shown_pdf))), p=self.shown_pdf)
    def from_examples(self, all_examples):
        eg_hidden = len(all_examples) - self.eg_shown
        return ConstantBatch(self.eg_shown, eg_hidden, self.is_shuffled).from_examples(all_examples)
    def __repr__(self):
        return 'RandomBatch{}'.format((self.is_shuffled,) + self.shown_pdf)


class SubsetBatch(object):
    def __init__(self, number_examples, shown_examples_list):
        self.is_shuffled = False
        self.number_examples = number_examples
        self.shown_examples_list = shown_examples_list
    def randomize(self):
        pass
    def from_examples(self, all_examples):
        assert len(all_examples) == self.number_examples, str((len(all_examples), self.number_examples))
        train_eg = [all_examples[i]
            for i in range(self.number_examples)
            if i in self.shown_examples_list]
        val_only_eg = [all_examples[i]
            for i in range(self.number_examples)
            if i not in self.shown_examples_list]
        return train_eg, val_only_eg
    def __repr__(self):
        return 'SubsetBatch({}, {})'.format(self.number_examples, self.shown_examples_list)


def collate_wrapper(collate_fn, batch_creator):
    def wrapped(batch, *args, **kwargs):
        batch_creator.randomize()
        for item in batch:
            item.resplit_examples(batch_creator)
        return collate_fn(batch, *args, **kwargs)
    return wrapped
