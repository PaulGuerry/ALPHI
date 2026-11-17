"""Minimal stand-in for tqdm (no progress bars in the browser; progress goes to the log)."""


class tqdm:
    def __init__(self, iterable=None, total=None, desc=None, **kw):
        self.iterable, self.total, self.desc, self.n = iterable, total, desc, 0

    def __iter__(self):
        for x in self.iterable:
            yield x
            self.n += 1

    def update(self, n=1):
        self.n += n

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass
