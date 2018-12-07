import asyncio
import logging

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from typing import Dict
import dataset

from overrides import overrides

from story_untangling.dataset_readers.create_dataset import create_dataset_db, negative_sentence_sampler
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
    sentence_predictive_window : int, (optional, default=1)
        Reads as a sliding windows over sentences. How many sentences to predict.
    target_positive : bool, (optional, default=True)
        Whether to yield the target predictive context or not.
    target_negative : bool, (optional, default=True)
        Whether to yield a negative target context or not.
    dataset_path : str, (optional, default=./dataset-cache/)
        Path for caching the dataset features. Will reuse rather than rebuilding if it finds the SQLLite dataset.
    use_existing_cached_db : bool, (optional, default=True)
        Reused an existing cached database db if it exists or delete and recreate.
    db_discriminator : str (optional, default=def)
        Allow multiple databases to be kept separate rather than wiping over each other.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 sentence_context_window: int = 2,
                 sentence_predictive_window: int = 1,
                 target_positive: bool = True,
                 target_negative: bool = True,
                 dataset_path: str = "./dataset-cache/",
                 use_existing_cached_db: bool = True,
                 db_discriminator="def",

                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._sentence_context_window = sentence_context_window
        self._sentence_predictive_window = sentence_predictive_window
        self._target_positive: bool = target_positive
        self._target_negative: bool = target_negative
        self._sentence_splitter = SpacySentenceSplitter()
        self._dataset_path = dataset_path
        self._use_existing_cached_db = use_existing_cached_db
        self._db_discriminator = db_discriminator

    @overrides
    def _read(self, file_path):

        loop = asyncio.get_event_loop()
        dataset_db = loop.run_until_complete(
            create_dataset_db(self._dataset_path, self._db_discriminator, file_path, self._use_existing_cached_db))

        db = dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600})

        negative_sampler = negative_sentence_sampler(db)

        stories = db.query('SELECT * FROM story ORDER BY id')
        for story in stories:
            story_id = story["id"]
            # Id will be the same as the sentence num as they are inserted as a batch in sequence.
            sentences = [s for s in db.query(f'SELECT * FROM sentence WHERE story_id = {story_id} ORDER BY id')]
            if len(sentences) == 0:
                logging.warning(f"Story has no sentences: {story_id}")
                continue

            sentence_text = [s["text"] for s in sentences]

            for source_sequence, target_sequence, absolute_count, relative_count in dual_window(sentence_text,
                                                                                                context_size=self._sentence_context_window,
                                                                                                predictive_size=self._sentence_predictive_window,
                                                                                                num_of_sentences=story[
                                                                                                    "sentence_num"]):
                source_sequence = " ".join(source_sequence)
                logging.debug(f"Source: '{source_sequence}', Target: '{target_sequence}'")

                negative_sequence = None
                if self._target_negative:
                    negative_sequence = ""
                    for i in range(self._sentence_predictive_window):
                        negative_dict = next(negative_sampler)
                        negative_sequence += negative_dict["text"]

                metadata = {"story_id": story_id}

                yield self.text_to_instance(source_sequence, target_sequence, negative_sequence, metadata,
                                            absolute_count, relative_count)

    @overrides
    def text_to_instance(self, source_tokens: str, target_tokens: str = None,
                         target_negative_tokens: str = None, metadata: Dict[str, any] = None,
                         absolute_position: int = 0, relative_position: float = 0.0) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        field_dict = {}

        def tokenize(tokens, tokenizer, indexer):
            tokenized_source = tokenizer(tokens)
            if self._source_add_start_token:
                tokenized_source.insert(0, Token(START_SYMBOL))
            tokenized_source.append(Token(END_SYMBOL))
            token_field = TextField(tokenized_source, indexer)
            return token_field

        field_dict['source_tokens'] = tokenize(source_tokens, self._source_tokenizer.tokenize,
                                               self._source_token_indexers)
        if target_tokens is not None:
            field_dict['target_tokens'] = tokenize(target_tokens, self._target_tokenizer.tokenize,
                                                   self._target_token_indexers)
        if target_negative_tokens is not None:
            field_dict['negative_tokens'] = tokenize(target_negative_tokens, self._target_tokenizer.tokenize,
                                                     self._target_token_indexers)

        # Wrap in an array there isn't a single value scalar field.
        field_dict["absolute_position"] = ArrayField([absolute_position])
        field_dict["relative_position"] = ArrayField([relative_position])

        field_dict["metadata"] = MetadataField(metadata)

        return Instance(field_dict)


