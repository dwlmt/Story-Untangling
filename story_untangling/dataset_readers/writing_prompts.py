import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
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
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter, SentenceSplitter
from aiofile import AIOFile, LineReader
from typing import Dict, Tuple, List, Any
import aiofile
import dataset
from dataset import Database

from overrides import overrides
from sqlalchemy import Index

from story_untangling.dataset_readers.dataset_utils import dual_window

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import multiprocessing as mp


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
            sentence_lengths = [s["sentence_len"] for s in sentences]

            for source_sequence, target_sequence in dual_window(sentence_text, size=self._sentence_context_window):
                source_sequence = " ".join(source_sequence)
                logging.debug(f"Source: '{source_sequence}', Target: '{target_sequence}'")

                negative_tokens = None
                if self._target_negative:
                    negative_dict = next(negative_sampler)
                    negative_tokens = negative_dict["text"]

                yield self.text_to_instance(source_sequence, target_sequence, negative_tokens)

    @overrides
    def text_to_instance(self, source_tokens: str, target_tokens: str = None,
                         target_negative_tokens: str = None, ) -> Instance:  # type: ignore
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
            field_dict['target_tokens'] = tokenize(source_tokens, self._target_tokenizer.tokenize,
                                                   self._target_token_indexers)
        if target_negative_tokens is not None:
            field_dict['negative_tokens'] = tokenize(source_tokens, self._target_tokenizer.tokenize,
                                                     self._target_token_indexers)

        return Instance(field_dict)


async def create_dataset_db(dataset_path: str, db_discriminator: str, file_path: str, use_existing_database=True,
                            sentence_splitter: SentenceSplitter = SpacySentenceSplitter(), batch_size: int = 100,
                            max_workers: int = 4) -> str:
    file_name = os.path.basename(file_path)
    database_file = f"{dataset_path}/{file_name}_{db_discriminator}.db"
    dataset_db = f"sqlite:///{database_file}"
    logging.info(f"Cached dataset path: {dataset_db}")

    # Create dir
    try:
        os.makedirs(dataset_path)
    except OSError:
        pass

    # Remove database if it shouldn't be reused.
    if not use_existing_database:
        try:
            os.remove(database_file)
        except OSError:
            pass

    if not Path(dataset_db).is_file():

        loop = asyncio.get_event_loop()

        f"{dataset_db}?mode=ro&cache=shared"
        db = dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600})
        db.commit()
        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            async for lines, story_nums in chunk_stories_from_file(file_path, batch_size=batch_size):
                story_sentences = sentence_splitter.batch_split_sentences(lines)
                tasks.append(
                    loop.run_in_executor(executor, SaveStoryToDatabase(dataset_db), story_sentences, story_nums))
            for task in tasks:
                story_ids = await task
                logger.info(f"Saved stories to db with ids: {story_ids}")
                # Add indices to key fields.

            try:
                db.query('CREATE INDEX idx_story_id on sentence(story_id)')
            except:
                pass  # Because of the aysnc/

    return dataset_db


async def chunk_stories_from_file(file: str, batch_size: int = 100) -> Tuple[List[str], List[int]]:
    """ Async yield batches of stories that are line separated/
    """
    line_count = 1
    lines = []
    story_nums = []
    async with AIOFile(file, mode="rb") as f:
        async for line in LineReader(f):
            line = line.decode('utf-8', errors="ignore")
            line = line.replace("<newline>", "")
            lines.append(line)
            story_nums.append(line_count)
            if len(lines) == batch_size:
                yield lines, story_nums
                lines = []
                story_nums = []

    yield lines, story_nums


class SaveStoryToDatabase:
    def __init__(self, dataset_db):
        self._dataset_db = dataset_db

    def __call__(self, story_sentences: List[str], story_nums: List[int]) -> List[int]:
        story_ids = []
        for sentences, story_num in zip(story_sentences, story_nums):
            db = dataset.connect(self._dataset_db, engine_kwargs={"pool_recycle": 3600})
            db.begin()
            try:
                story_table = db['story']
                sentence_table = db['sentence']
                story = dict(story_num=story_num)
                story_id = story_table.insert(story)
                sentences_to_save = []

                total_story_tokens = 0
                for i, sent in enumerate(sentences):
                    sentence_len = len(sent)
                    total_story_tokens += sentence_len
                    sentences_to_save.append(
                        dict(sentence_num=i, text=sent, sentence_len=sentence_len, story_id=story_id))
                sentence_table.insert_many(sentences_to_save)

                story_table.update(dict(sentence_num=len(sentences), tokens_num=total_story_tokens, id=story_id),
                                   ['id'])
                db.commit()
                story_ids.append(story_id)
            except:
                db.rollback()
        return story_ids


def negative_sentence_sampler(db: Database) -> Dict[str, Any]:
    while True:
        random_sentences = db.query(f'SELECT * FROM sentence ORDER BY RANDOM()')
        for sentence in random_sentences:
            yield sentence
