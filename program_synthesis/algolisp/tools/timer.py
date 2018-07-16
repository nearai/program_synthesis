import collections
import time


def _print_tags(order, tags, counts={}):
    result = ', '.join([
        "%s=%.5f (%d)" % (tag, value, counts.get(tag, 0))
        for tag, value in tags.items()
    ])
    print(result)


class Timer(object):
    def __init__(self):
        self.tag_order = []
        self.tags = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)
        self.last_time = time.time()

    def reset(self):
        self.last_time = time.time()

    def display(self):
        _print_tags(self.tag_order, self.tags, self.counts)

    def tag(self, tag):
        if tag not in self.tags:
            self.tag_order.append(tag)
        self.tags[tag] = time.time() - self.last_time
        self.last_time = time.time()

    def acc(self, tag):
        if tag not in self.tag_order:
            self.tag_order.append(tag)
        self.tags[tag] += time.time() - self.last_time
        self.counts[tag] += 1
        self.last_time = time.time()


_TIMERS = {}


def timer(name):
    global _TIMERS
    if name not in _TIMERS:
        _TIMERS[name] = Timer()
    return _TIMERS[name]
