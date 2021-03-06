from collections import OrderedDict
from typing import Dict, TypeVar, Generic, Optional

import torch


def random_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    d = torch.distributions.Categorical(logits=logits / temperature)
    return d.sample().item()


K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int, default_value=None):
        self._capacity = capacity
        self._cache: Dict[K, V] = OrderedDict()
        self._default_value = default_value

    def __getitem__(self, key: K) -> Optional[V]:
        if self._capacity == 0:
            return self._default_value
        try:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        except KeyError:
            return self._default_value

    def __setitem__(self, key: K, value: V) -> None:
        if self._capacity == 0:
            return

        try:
            self._cache.pop(key)
        except KeyError:
            if len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
        self._cache[key] = value
