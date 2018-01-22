import argparse
import sys
import collections


def code_stat(code_tree, language):
    if language == 'lisp':
        return lisp_code_stat(code_tree)
    else:
        raise ValueError("Unknown language: %s" % language)


def lisp_code_stat(code_tree):
    if isinstance(code_tree, basestring):
        return 1, 1
    depth, count = 0, 1
    for x in code_tree[1:]:
        d, c = lisp_code_stat(x)
        depth = max(depth, d)
        count += c
    return depth + 1, count


def test_cardinality(tests):
    test_outputs = set()
    for test in tests:
        test_outputs.add(str(test['output']))
    return len(test_outputs)


class DatasetStats(object):

    def __init__(self, args):
        self.args = args
        self.text_map = set()
        self.code_map = set()
        self.stats = {
            'total_text_len': 0, 'total_code_count': 0, 'total_code_depth': 0,
            'total_arg_count': 0, 'total_func_count': 0,
            'total': 0, 'total_same_input_tests': 0, 'total_same_other_tests': 0,
            'min_code_len': 0, 'max_code_len': 0, 'min_text_len': 0, 'max_text_len': 0}
        self.task_types_stats = collections.defaultdict(int)
        self.tags_stats = collections.defaultdict(int)

    def update(self, example):
        v = len(example.text)
        self.stats['total_text_len'] += v
        self.stats['max_text_len'] = max(self.stats['max_text_len'], v)
        if self.stats['min_text_len'] > 0:
            self.stats['min_text_len'] = min(self.stats['min_text_len'], v)
        else:
            self.stats['min_text_len'] = v
        if example.code_tree:
            if example.language == 'lisp':
                depth, count = lisp_code_stat(example.code_tree)
                self.stats['total_code_depth'] += depth
                self.stats['total_code_count'] += count
                self.stats['total_func_count'] += len(example.funcs)
        if example.code_sequence:
            v = len(example.code_sequence)
            self.stats['total_code_count'] += v
            self.stats['max_code_len'] = max(self.stats['max_code_len'], v)
            if self.stats['min_code_len'] > 0:
                self.stats['min_code_len'] = min(self.stats['min_code_len'], v)
            else:
                self.stats['min_code_len'] = v
        self.stats['total_arg_count'] += len(example.schema.args)
        self.stats['total'] += 1
        self.text_map.add(str(example.text))
        self.code_map.add(str(example.code_sequence))
        self.stats['total_same_input_tests'] += 1 if test_cardinality(example.input_tests) == 1 else 0
        self.stats['total_same_other_tests'] += 1 if test_cardinality(example.tests) == 1 else 0
        for name in example.task_types:
            self.task_types_stats[name] += 1
        for name in example.tags:
            self.tags_stats[name] += 1

    def display(self):
        print(
            ("Total: %d\n" + 
            "Avg text length: %.2f (%d, %d), Avg code count: %.2f (%d, %d), Avg code depth: %.2f\n" +
            "Avg arg count: %.2f, Avg func count: %.2f\n" + 
            "Unique texts: %d, Unique codes: %d\n" + 
            "Same input tests: %d, same other tests: %d") % (
                self.stats['total'],
                float(self.stats['total_text_len']) / self.stats['total'],
                self.stats['min_text_len'], self.stats['max_text_len'],
                float(self.stats['total_code_count']) / self.stats['total'],
                self.stats['min_code_len'], self.stats['max_code_len'],
                float(self.stats['total_code_depth']) / self.stats['total'],
                float(self.stats['total_arg_count']) / self.stats['total'],
                float(self.stats['total_func_count']) / self.stats['total'],
                len(self.text_map), len(self.code_map),
                self.stats['total_same_input_tests'],
                self.stats['total_same_other_tests']
            )
        )
        if self.args.show_tags:
            for name, value in self.task_types_stats.iteritems():
                print("%s: %d" % (name, value))
            for name, value in self.tags_stats.iteritems():
                print("%s: %d" % (name, value))


def report_stats(args, dataset):
    ds = DatasetStats(args)
    for example in dataset.data:
        ds.update(example)
    ds.display()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Statistics')
    parser.add_argument('--dataset', type=str, default='algolisp')
    parser.add_argument('--dataset_max_size', type=int, default=0)
    parser.add_argument('--dataset_max_code_length', type=int, default=0)
    parser.add_argument('--show-tags', action='store_true', default=False)
    parser.add_argument('--vocab_min_freq', type=int, default=50)
    args, _ = parser.parse_known_args(sys.argv)

    import dataset
    args.batch_size = 1
    train_dataset, _ = dataset.get_dataset(args)
    report_stats(args, train_dataset)
