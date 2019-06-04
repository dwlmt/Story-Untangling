import asyncio
import logging
import textwrap
from typing import Dict, List, Union, Any

import dataset
import enchant
import more_itertools
import nltk
import torch
from PyDictionary import PyDictionary
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from nostril import nonsense
from overrides import overrides
from spacy.lang.en import STOP_WORDS

from story_untangling.dataset_readers.dataset_features import create_dataset_db

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from itertools import groupby
from string import punctuation

punc = set(punctuation) - set('.')


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
                 min_story_sentences: int = 5,
                 max_story_sentences: int = 500,
                 truncate_sequence_length: int = 50,
                 max_avg_length_per_word = 8,
                 max_word_length = 25,
                 min_check_word_length=8,
                 story_chunking: int = 50,
                 ner_model: str = None,
                 coreference_model: str = None,
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
        self._max_character_length = self._truncate_sequence_length * max_avg_length_per_word
        self._max_word_length = max_word_length
        self._min_check_word_length = min_check_word_length
        self._truncate_sequences = (truncate_sequence_length != 0)
        self._story_chunking = story_chunking
        self._ner_model = ner_model
        self._coreference_model = coreference_model

        self._cuda_device = cuda_device

        self._allowed_tokens = set([])
        self._tried_tokens = set([])

        for t in nltk.corpus.stopwords.words('english'):
            self._allowed_tokens.add(t)
            self._tried_tokens.add(t)

        for t in STOP_WORDS:
            self._allowed_tokens.add(t)
            self._tried_tokens.add(t)

        for t in punctuation:
            self._allowed_tokens.add(t)
            self._tried_tokens.add(t)

        self._py_dictionary = PyDictionary()
        self._enchant_dict_us = enchant.Dict("en_US")
        self._enchant_dict_uk = enchant.Dict("en_UK")

        self._seen_datasets = set()

        self._tried_to_insert = []
        self._allowed_to_insert = []

    @overrides
    def _read(self, file_path):

        batch_num = 0

        tensors = self.block_memory()
        self.unblock_memory(tensors)

        loop = asyncio.get_event_loop()
        dataset_db = loop.run_until_complete(
            create_dataset_db(dataset_path=self._dataset_path, db_discriminator=self._db_discriminator,
                              file_path=file_path, use_existing_database=self._use_existing_cached_db,
                              ner_model=self._ner_model, coreference_model=self._coreference_model,
                              cuda_device=self._cuda_device))

        self._dataset_db = dataset_db

        with dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600}) as db:

            if not file_path in self._seen_datasets:
                self._seen_datasets.add(file_path)
                allowed_tokens = db.get_table("allowed_tokens")
                for t in allowed_tokens:
                    self._allowed_tokens.add(t["token"])
                print(f"Starting Allowed Tokens Size: {len(self._allowed_tokens)}")
                tried_tokens = db.get_table("tried_tokens")
                for t in tried_tokens:
                    self._tried_tokens.add(t["token"])
                print(f"Starting Tried Tokens Size: {len(self._tried_tokens)}")

            # Randomize the order of the stories. With repeated epochs and lazy dataloaders will produce different negative examples each epoch.
            stories = db.query(
                f'SELECT * FROM story WHERE sentence_num >= {self._min_story_sentences} '
                f'AND sentence_num <= {self._max_story_sentences} ORDER BY random()')

            for i, story in enumerate(stories):

                story_id = story["id"]

                # Id will be the same as the sentence num as they are inserted as a batch in sequence.
                sentences = [s for s in db.query(
                    f'SELECT * FROM sentence INNER JOIN sentence_lang on sentence.id = sentence_lang.sentence_id '
                    f'WHERE sentence.story_id = {story_id} and sentence_lang.lang = "en" '
                    f'and sentence_lang.nonsense = false and sentence_lang.ascii_chars=true ORDER BY id')]

                if sentences is None or len(sentences) == 0:
                    print(f"Skip story {story_id} with no valid sentences")
                    continue

                for sentence_batch in list(more_itertools.chunked(sentences, self._story_chunking)):
                    # Filter out non English and gibberish sentences.

                    yield self.text_to_instance(sentence_batch, story, db=db, batch_num=batch_num)
                    batch_num += 1

        with dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600}) as db:
            self._insert_tried_and_allowed_tokens(db)

        disgarded_tokens = self._tried_tokens.difference(self._allowed_tokens)
        print(f"Disgarded tokens, num {len(disgarded_tokens)}: {disgarded_tokens}")

    def unblock_memory(self, tensors):
        if tensors is None or len(tensors) == 0:
            return

        for t in tensors:
            t = t.cpu()
            del t
        del tensors
        #torch.cuda.empty_cache()

    def block_memory(self):
        ''' Blocks GPU memory unless 1GB is already being used. The random memory tensor is 8GB.
        '''
        tensors = []
        with torch.no_grad():
            if isinstance(self._cuda_device, int):
                memory_used = max(torch.cuda.memory_allocated(self._cuda_device),torch.cuda.memory_cached(device=self._cuda_device))
                if memory_used < 1e+9:
                    tensors.append(torch.rand(int(20e8), device=self._cuda_device))
            else:
                for cuda_device in self._cuda_device:
                    memory_used = max(torch.cuda.memory_allocated(cuda_device),torch.cuda.memory_cached(device=cuda_device))
                    if memory_used < 1e+9:
                        tensors.append(torch.rand(int(20e8), device=cuda_device))

            return tensors

    def _insert_tried_and_allowed_tokens(self, db):
        try:
            if len(self._tried_to_insert) > 0:
                db["tried_tokens"].insert_many(self._tried_to_insert)
                print(f"Tried tokens inserted: {len(self._tried_to_insert)}")
                self._tried_to_insert = []
            if len(self._allowed_to_insert) > 0:
                db["allowed_tokens"].insert_many(self._allowed_to_insert)
                print(f"Allowed tokens inserted: {len(self._allowed_to_insert)}")
                self._allowed_to_insert = []

        except:
            print(f"Couldn't insert {self._tried_to_insert}, {self._allowed_to_insert}")

    def sample_random_sentences(self, n, story_id=None, sentence_id=None):
        ''' Sample n random sentences from the corpus.
            If sentence and story id specified then always choose the sentences real successor as well.
        '''
        sentence_batch = []
        story_batch = []

        if sentence_id:
            n = n - 1
        with dataset.connect(self._dataset_db, engine_kwargs={"pool_recycle": 3600}) as db:
            sentences = list(db.query(
                f'SELECT * FROM sentence INNER JOIN sentence_lang on sentence.id = sentence_lang.sentence_id '
                f'WHERE sentence_lang.lang = "en" '
                f'and sentence_lang.nonsense = false and sentence_lang.ascii_chars=true ORDER BY random() LIMIT {n}'))

            if sentence_id:
                next_sentence = db.query(
                    f'SELECT * FROM sentence INNER JOIN sentence_lang on sentence.id = sentence_lang.sentence_id '
                    f'WHERE sentence.id = {sentence_id + 1} and sentence.story_id={story_id} and sentence_lang.lang = "en" '
                    f'and sentence_lang.nonsense = false and sentence_lang.ascii_chars=true')

                for n in next_sentence:
                    # print(n)
                    sentences.append(n)

            for sentence in sentences:
                story = db.query(
                    f'SELECT * FROM story WHERE id={sentence["story_id"]}')

                for s in story:
                    story_batch.append(s)
                # The story fields are in the dict which is why sentences is parsed twice.
                sentence_batch.append(sentence)

        return self.text_to_instance(sentence_batch, story_batch, db)


    @overrides
    def text_to_instance(self,
                         sentences: Dict[str, Any], story: Dict[str, Any], db, batch_num=None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        field_dict = {}
        story_text_original = []
        story_text_fields = []
        sentence_lengths = []

        def tokenize(sentence, tokenizer, indexer):
            if isinstance(sentence, str):
                tokens = sentence
            else:
                tokens = sentence["text"]

            tokens = text_standardize(tokens)

            tokens = strip_repeating_punctuation(tokens)

            tokens = textwrap.shorten(tokens, width=self._max_character_length)

            tokenized_text = tokenizer(tokens)

            stripped_tokens = []

            for token in tokenized_text:
                token_text = token.lower_
                token_len = len(token)

                # Disable the main checker
                if token_len > self._max_word_length:
                    if len(token.pos_) > 0:
                        stripped_tokens.append(Token(token.pos_))
                elif token_len < self._min_check_word_length or token_text in self._allowed_tokens:
                    stripped_tokens.append(token)
                elif token_text not in self._tried_tokens:

                    lookup_tokens(token_text)

                    if token_text in self._allowed_tokens and len(token_text):
                        stripped_tokens.append(token)
                    else:
                        if len(token.pos_) > 0:
                            stripped_tokens.append(Token(token.pos_))
                        print(f"Rejected token: {token_text}")

            tokenized_text = stripped_tokens


            if self._add_start_end_token:
                tokenized_text.insert(0, Token(START_SYMBOL))
                tokenized_text.append(Token(END_SYMBOL))

            if len(tokenized_text) > self._truncate_sequence_length and self._truncate_sequences:
                tokenized_text = tokenized_text[:self._truncate_sequence_length]

            token_field = TextField(tokenized_text, indexer)

            return tokens, token_field

        def lookup_tokens(token_text):

            add_token = False
            if self._enchant_dict_us.check(token_text) or self._enchant_dict_uk.check(token_text):
                print("Enchant Dictionary", token_text)
                add_token = True
            elif self._py_dictionary.meaning(token_text) != None:
                print("Py Dictionary", token_text)
                add_token = True
            else:
                try:
                    if not nonsense(token_text):
                        print("Not Nonsense", token_text)
                        add_token = True
                except:
                    pass

            if add_token:
                self._allowed_tokens.add(token_text)
                self._allowed_to_insert.append({"token": token_text})

            self._tried_tokens.add(token_text)
            self._tried_to_insert.append({"token": token_text})

        def strip_repeating_punctuation(tokens):
            # Strip repeating characters.
            newtext = []
            for k, g in groupby(tokens):
                if k in punc:
                    newtext.append(k)
                else:
                    newtext.extend(g)
            tokens = ''.join(newtext)
            return tokens

        for i, sentence in enumerate(sentences, 1):
            sentence_text, sentence_text_field, = tokenize(sentence, self._tokenizer.tokenize,
                                                           self._token_indexers)

            sentence_length = sentence_text_field.sequence_length
            sentence_lengths.append(sentence_length)
            if sentence_length == 0:
                print("Empty Sentence")
                sentence_text_field = sentence_text_field.empty_field()
            story_text_original.append(sentence_text)
            story_text_fields.append(sentence_text_field)

        if len(story_text_fields) < self._story_chunking:
            text_field = TextField([], self._token_indexers).empty_field()
            story_text_fields.extend([text_field] * (self._story_chunking - len(story_text_fields)))

        if isinstance(story, list):
            story_id = [s["id"] for s in story]
            sentence_num = [s["sentence_num"] for s in story]
        else:
            story_id = story["id"]
            sentence_num = story["sentence_num"]

        metadata = {"story_id": story_id, "sentence_ids": [s["id"] for s in sentences],
                    "sentence_nums": [s["sentence_num"] for s in sentences],
                    "number_of_sentences": sentence_num}
        metadata["text"] = story_text_original
        metadata["sentence_lengths"] = sentence_lengths

        if batch_num is not None:
            metadata["batch_num"] = batch_num

        text_field = ListField(story_text_fields)
        field_dict['text'] = text_field
        field_dict["metadata"] = MetadataField(metadata)

        return Instance(field_dict)

    def create_temp_dataset(self, temp_db_location):
        self._dataset_path = temp_db_location
        self._use_existing_cached_db = False
