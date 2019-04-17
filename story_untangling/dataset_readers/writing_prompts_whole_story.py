import asyncio
import logging
from collections import OrderedDict
from time import sleep
from typing import Dict, List, Union, Any

import dataset
import more_itertools
import numpy
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph
from overrides import overrides

from story_untangling.dataset_readers.dataset_features import create_dataset_db, negative_sentence_sampler
from story_untangling.dataset_readers.dataset_utils import dual_window
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("writing_prompts_whole_story")
class WritingPromptsWholeStoryDatasetReader(DatasetReader):
    """
    This is derived from the Sequence to Sequence AllenNLP base class. Unlike normal sequence to sequence the source and
    target are sliding windows of context text and text to predict.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.

    add_start_end_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.

    dataset_path : str, (optional, default=./dataset-cache/)
        Path for caching the dataset features. Will reuse rather than rebuilding if it finds the SQLLite dataset.
    use_existing_cached_db : bool, (optional, default=True)
        Reused an existing cached database db if it exists or delete and recreate.
    db_discriminator : str (optional, default=def)
        Allow multiple databases to be kept separate rather than wiping over each other.
    min_story_sentences : int, (optional, default=0)
        Min number of sentences a story must have to be included.
    max_story_sentences : int, (optional, default=1000000)
        Max number of sentences a story must have to be included.
    truncate_sequence_length : int, (optional, default=100)
        Sentences longer than this will be truncated (cutting starting from the end).
        0 indicates length is unlimited. Value must be greater than or equal to 0.
    story_embedding : bool, (optional, default=False)
        Provide a single vector per story to represent changes as it progresses through each sentence.
     : bool, (optional, default=False)
    story_chunking : int, (optional, default=100)
        Max sentences to chunk together in the story.
    cuda_device : List[Int] (optional, default=-1)
        List of CUDA devices. This is needed in cases such as NER and coreferencing where preprocessing benefits from CUDA.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_end_token: bool = False,
                 dataset_path: str = "./dataset-cache/",
                 use_existing_cached_db: bool = True,
                 db_discriminator="def",
                 min_story_sentences: int = 0,
                 max_story_sentences: int = 10 * 6,
                 truncate_sequence_length: int = 60,
                 story_chunking: int = 100,
                 cuda_device: Union[List[int], int] = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._add_start_end_token = add_start_end_token

        self._dataset_path = dataset_path
        self._use_existing_cached_db = use_existing_cached_db
        self._db_discriminator = db_discriminator
        self._min_story_sentences = min_story_sentences
        self._max_story_sentences = max_story_sentences
        self._truncate_sequence_length = truncate_sequence_length
        self._truncate_sequences = (truncate_sequence_length != 0)
        self._story_chunking = story_chunking

        self._cuda_device = cuda_device

    @overrides
    def _read(self, file_path):

        loop = asyncio.get_event_loop()
        dataset_db = loop.run_until_complete(
            create_dataset_db(dataset_path=self._dataset_path, db_discriminator=self._db_discriminator,
                              file_path=file_path, use_existing_database=self._use_existing_cached_db,
                              truncate_sequence_length=self._truncate_sequence_length,
                              cuda_device=self._cuda_device))

        db = dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600})

        # Randomize the order of the stories. With repeated epochs and lazy dataloaders will produce different negative examples each epoch.
        stories = db.query(
            f'SELECT * FROM story  WHERE sentence_num >= {self._min_story_sentences} '
            f'AND sentence_num <= {self._max_story_sentences} ORDER BY random()')

        for story in stories:

                story_id = story["id"]

                # Id will be the same as the sentence num as they are inserted as a batch in sequence.
                sentences = [s for s in db.query(f'SELECT * FROM sentence WHERE story_id = {story_id} ORDER BY id')]

                for sentence_batch in list(more_itertools.chunked(sentences, self._story_chunking)):

                    yield self.text_to_instance(sentence_batch, story)

    @overrides
    def text_to_instance(self,
                         sentences: Dict[str, Any], story: Dict[str, Any]) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        field_dict = {}
        story_text_original = []
        story_text_fields = []

        def tokenize(sentence, tokenizer, indexer):
            if isinstance(sentence, str):
                tokens = sentence
            else:
                tokens = sentence["text"]

            tokens = text_standardize(tokens)

            tokenized_text = tokenizer(tokens)

            if self._add_start_end_token:
                tokenized_text.insert(0, Token(START_SYMBOL))
                tokenized_text.append(Token(END_SYMBOL))

            if len(tokenized_text) > self._truncate_sequence_length and self._truncate_sequences:
                tokenized_text = tokenized_text[:self._truncate_sequence_length]

            token_field = TextField(tokenized_text, indexer)

            if len(tokenized_text) == 0:
                token_field = token_field.empty_field()

            return tokens, token_field

        for i, sentence in enumerate(sentences, 1):
            sentence_text, sentence_text_field, = tokenize(sentence, self._tokenizer.tokenize,
                                                           self._token_indexers)
            story_text_original.append(sentence_text)
            story_text_fields.append(sentence_text_field)

        if len(story_text_fields) < self._story_chunking:
            text_field = TextField([], self._token_indexers).empty_field()
            story_text_fields.extend([text_field] * (self._story_chunking - len(story_text_fields)))

        metadata = {"story_id": story["id"], "number_of_sentences": story["sentence_num"]}
        metadata["text"] = story_text_original

        text_field = ListField(story_text_fields)
        field_dict['text'] = text_field
        field_dict["metadata"] = MetadataField(metadata)

        return Instance(field_dict)


    def create_temp_dataset(self, temp_db_location):
        self._dataset_path = temp_db_location
        self._use_existing_cached_db = False
