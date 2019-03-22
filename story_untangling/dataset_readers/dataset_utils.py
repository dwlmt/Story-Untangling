from typing import Tuple, Iterable

from more_itertools import windowed


def dual_window(seq: Iterable[str], context_size: int = 2, predictive_size: int = 1, num_of_sentences: int = 1, step: int = 1) -> \
        Tuple[Iterable[str], Iterable[str], int, float]:
    """ Create a sliding window except the last value is returned in a separate tuple so can be used for prediction.
    """
    context_size = min(context_size + predictive_size, len(seq))
    for i, slice in enumerate(windowed(seq, context_size, step = step), start=1):
        s = len(slice)
        if s == context_size:
            source_win, target_win, abs_pos, rel_pos = slice[0: s - predictive_size], slice[-predictive_size], i, i / float(num_of_sentences)

            # Wrap single values in a list for more consistent handling later.
            if isinstance(source_win, int):
                source_win = [source_win]
            if isinstance(target_win, int):
                target_win = [target_win]

            yield source_win, target_win, abs_pos, rel_pos
