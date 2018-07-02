import collections
import time


def _print_tags(tags, counts={}):
    result = ', '.join([
        "%s=%.5f (%d)" % (tag, value, counts.get(tag, 0))
        for tag, value in tags.iteritems()
    ])
    print(result)


class Timer(object):
    def __init__(self):
        self.tags = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)
        self.last_time = time.time()

    def reset(self):
        self.last_time = time.time()

    def display(self):
        _print_tags(self.tags, self.counts)

    def tag(self, tag):
        self.tags[tag] = time.time() - self.last_time
        self.last_time = time.time()

    def acc(self, tag):
        self.tags[tag] += time.time() - self.last_time
        self.counts[tag] += 1
        self.last_time = time.time()


_TIMERS = {}


def timer(name):
    global _TIMERS
    if name not in _TIMERS:
        _TIMERS[name] = Timer()
    return _TIMERS[name]
