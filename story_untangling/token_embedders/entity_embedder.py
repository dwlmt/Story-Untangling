import io
import tarfile
import zipfile
import re
import logging
import warnings
import itertools
from collections import defaultdict, deque
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple

from allennlp.modules import Embedding
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from overrides import overrides
import numpy
import torch
from torch.nn.functional import embedding

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import get_file_extension, cached_path
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("entity_embedding")
class EntityEmbedding(Embedding):
    """
    An updatable embedding modeled on EntityNLM.
    ----------

    num_embeddings : int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    norm_type : float, (optional, default=2):
        The p of the p-norm to compute for the max_norm option
    keep_history : int, (optional, default=0)
        If a history of the embeddings should be kept. If positive then the max number of updates to keep.
    Returns
    -------
    An Embedding module.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 norm_type: float = 2.,
                 keep_history: int = None
                 ) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_index = padding_index
        self.norm_type = norm_type
        self.keep_history = keep_history

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            torch.nn.init.normal_(self.weight)
            self.weight.data = torch.nn.functional.normalize(input=self.weight, p=self.norm_type, dim=1)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        # Regardless of history keep the first.
        self.embeddings_initial = {i: e.clone().detach().cpu() for i, e in enumerate(self.weight.data)}

        self.output_dim = embedding_dim

        self.embeddings_history = defaultdict(lambda: deque(maxlen=self.keep_history))


    @overrides
    def forward(self, inputs):

        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        embedded = embedding(inputs, self.weight, padding_idx=self.padding_index)

        embedded = util.uncombine_initial_dims(embedded, original_size)

        return embedded

    def update(self, indices, updated_entities):

        for i, ent in zip(indices, updated_entities):

            self.weight.data[i, :] = ent

            if self.keep_history > 0:
                # Separate from computational graph and transfer to CPU so as not to run out of memory.
                self.embeddings_history[i.item()].append(ent.clone().detach().cpu())

                print("length", i, len(self.embeddings_history[i.item()]))

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':  # type: ignore
        """ Construct from parameters.
        """
        # pylint: disable=arguments-differ
        num_embeddings = params.pop_int('num_embeddings', None)
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop("pretrained_file", None)
        padding_index = params.pop_int('padding_index', None)
        norm_type = params.pop_float('norm_type', 2.)
        keep_history = params.pop_int('keep_history', 0)

        if pretrained_file:
            weight = _read_pretrained_embeddings_file(pretrained_file,
                                                      embedding_dim,
                                                      vocab,
                                                      vocab_namespace)
        else:
            weight = None

        params.assert_empty(cls.__name__)

        return cls(num_embeddings=num_embeddings, weight=weight, embedding_dim=embedding_dim,
                   padding_index=padding_index,
                   norm_type=norm_type, keep_history=keep_history)
