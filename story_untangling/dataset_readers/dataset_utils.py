from itertools import islice, tee

def window(it, size=2):
    """ A single step sliding window over an iterable.
    """
    yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(it, size))])

def dual_window(seq, size=2):
    """ Create a sliding window except the last value is returned in a separate tuple so can be used for prediction.
    """
    size = min(size+1, len(seq))
    for slice in window(seq, size=size):
        s = len(slice)
        if s > 1:
            yield slice[0: s-1], slice[-1]
        else:
            yield slice, None

