import json
import logging
from typing import Dict

import spacy
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter, SpacyWordSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides

from story_untangling.dataset_readers.dataset_utils import dual_window

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("writing_prompts")
class WritingPromptsDatasetReader(DatasetReader):
    """
    This is derived from the Sequence to Sequence AllenNLP base class. Unlike normal sequence to sequence the source and
    target are sliding windows of context text and text to predict.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    sentence_context_window : int, (optional, default=2)
        Reads as a sliding windows over sentences. How many sentences to use as a context for each sentence to predict.
    target_positive : bool, (optional, default=True)
        Whether to yield the target predictive context or not.
    target_negative : bool, (optional, default=True)
        Whether to yield a negative target context or not.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 sentence_context_window: int = 2,
                 target_positive: bool = True,
                 target_negative: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._sentence_context_window = sentence_context_window
        self._target_positive: bool = target_positive
        self._target_negative: bool = target_negative
        self._sentence_splitter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(f"WritingPromptsDatasetReader: Reading instances from lines in file at: {file_path}", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line = line.replace("<newline>","") # The newline isn't relevant, it's just Reddit formatting.
                sentences = self._sentence_splitter.split_sentences(line)
                for source_sequence, target_sequence in dual_window(sentences, size=self._sentence_context_window):
                    source_sequence = " ".join(source_sequence)
                    logging.debug(f"Source: '{source_sequence}', Target: '{target_sequence}'")
                    yield self.text_to_instance(source_sequence, target_sequence)



    @overrides
    def text_to_instance(self, source_tokens: str, target_tokens: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_tokens)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_tokens is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_tokens)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})