from itertools import islice, tee

from typing import List, Tuple, Iterable


def window(it: Iterable[str], size: int = 2):
    """ A single step sliding window over an iterable.
    """
    yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(it, size))])


def dual_window(seq: Iterable[str], context_size: int = 2, predictive_size: int = 1, num_of_sentences: int = 1) -> \
        Tuple[Iterable[str], Iterable[str], int, float]:
    """ Create a sliding window except the last value is returned in a separate tuple so can be used for prediction.
    """
    context_size = min(context_size + predictive_size, len(seq))
    for i, slice in enumerate(window(seq, size=context_size), start=1):
        s = len(slice)
        if s > predictive_size:
            yield slice[0: s - 1], slice[-predictive_size], i, i / float(num_of_sentences)
        else:
            yield slice, None, i, i / float(num_of_sentences)
