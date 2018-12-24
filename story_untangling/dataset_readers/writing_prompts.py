import asyncio
import logging
from typing import Dict, List, Union

import dataset
import numpy
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides

from story_untangling.dataset_readers.dataset_features import create_dataset_db, negative_sentence_sampler
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
    target_negative : bool, (optional, default=True)
        Whether to yield a negative target context or not.
    dataset_path : str, (optional, default=./dataset-cache/)
        Path for caching the dataset features. Will reuse rather than rebuilding if it finds the SQLLite dataset.
    use_existing_cached_db : bool, (optional, default=True)
        Reused an existing cached database db if it exists or delete and recreate.
    db_discriminator : str (optional, default=def)
        Allow multiple databases to be kept separate rather than wiping over each other.
    save_sentiment: bool (optional, default=True)
        Whether to save sentence level sentiment when creating the dataset.
    ner_model : str (optional, default=None)
        AllenNLP NER model to run for features.
    coreference_model : str (optional, default=None)
        AllenNLP Coreference model.
    min_story_sentences : int, (optional, default=0)
        Min number of sentences a story must have to be included.
    max_story_sentences : int, (optional, default=1000000)
        Max number of sentences a story must have to be included.
    positional_features : bool, (optional, default=True)
        Should encode positional features in the source.
    cuda_device : List[Int] (optional, default=-1)
        List of CUDA devices. This is needed in cases such as NER and coreferencing where preprocessing benefits from CUDA.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_end_token: bool = True,
                 sentence_context_window: int = 2,
                 sentence_predictive_window: int = 1,
                 target_negative: bool = True,
                 dataset_path: str = "./dataset-cache/",
                 use_existing_cached_db: bool = True,
                 db_discriminator="def",
                 save_sentiment: bool = True,
                 ner_model: str = None,
                 coreference_model: str = None,
                 min_story_sentences: int = 0,
                 max_story_sentences: int = 10 * 6,
                 positional_features: bool = True,
                 cuda_device: Union[List[int], int] = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._add_start_end_token = add_start_end_token
        self._sentence_context_window = sentence_context_window
        self._sentence_predictive_window = sentence_predictive_window
        self._target_negative: bool = target_negative
        self._sentence_splitter = SpacySentenceSplitter()
        self._dataset_path = dataset_path
        self._use_existing_cached_db = use_existing_cached_db
        self._db_discriminator = db_discriminator
        self._save_sentiment = save_sentiment
        self._ner_model = ner_model
        self._coreference_model = coreference_model
        self._min_story_sentences = min_story_sentences
        self._max_story_sentences = max_story_sentences
        self._positional_features = positional_features
        self._cuda_device = cuda_device

    @overrides
    def _read(self, file_path):

        loop = asyncio.get_event_loop()
        dataset_db = loop.run_until_complete(
            create_dataset_db(dataset_path=self._dataset_path, db_discriminator=self._db_discriminator,
                              should_save_sentiment=self._save_sentiment,
                              file_path=file_path, use_existing_database=self._use_existing_cached_db,
                              ner_model=self._ner_model,
                              coreference_model=self._coreference_model,
                              cuda_device=self._cuda_device))

        db = dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600})

        negative_sampler = negative_sentence_sampler(db)

        stories = db.query(
            f'SELECT * FROM story  WHERE sentence_num >= {self._min_story_sentences} '
            f'AND sentence_num <= {self._max_story_sentences} ORDER BY id')
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
                        negative_sequence += " " + negative_dict["text"]

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
            if self._add_start_end_token:
                tokenized_source.insert(0, Token(START_SYMBOL))
                tokenized_source.append(Token(END_SYMBOL))

            token_field = TextField(tokenized_source, indexer)

            if len(tokenized_source) == 0:
                token_field = token_field.empty_field()
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
        source_features = []

        # It only makes sense to include in the source features as otherwise it gives away the correct answers.
        if self._positional_features:
            source_features.append(absolute_position)
            source_features.append(relative_position)
        if len(source_features) > 0:
            field_dict["source_features"] = ArrayField(numpy.array(source_features))

        field_dict["metadata"] = MetadataField(metadata)

        return Instance(field_dict)
