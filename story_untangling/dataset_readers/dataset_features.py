import asyncio
import copy
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import dataset
import more_itertools
import nltk
from aiofile import AIOFile, LineReader
from allennlp.data.tokenizers import SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.models import Model
from allennlp.predictors import Predictor
from dataset import Database
from nltk.sentiment import SentimentIntensityAnalyzer
from nostril import nonsense
from textblob import TextBlob
from whatthelang import WhatTheLang

nltk.download('vader_lexicon')


def isAscii(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 1000, "check_same_thread": False}}


async def create_dataset_db(dataset_path: str, db_discriminator: str, file_path: str, use_existing_database=True,
                            sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                            should_save_sentiment: bool = True,
                            ner_model: str = True,
                            coreference_model: str = True,
                            batch_size: int = 100,
                            max_workers: int = 16,
                            marked_sentences=False,
                            cuda_device: Union[List[int], int] = None) -> str:
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

    if not Path(database_file).is_file():

        loop = asyncio.get_event_loop()

        with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

            # Create the main tables and columns that need indexing.
            story_table = db.create_table('story')
            story_table.create_column('story_num', db.types.integer)
            story_table.create_index(['story_num'])

            sentence_table = db.create_table('sentence')
            sentence_table.create_column('story_id', db.types.bigint)
            sentence_table.create_column('sentence_num', db.types.integer)
            sentence_table.create_column('sentence_len', db.types.integer)
            sentence_table.create_column('start_span', db.types.integer)
            sentence_table.create_column('end_span', db.types.integer)
            # Indices created at the beginning as creating them later causing other processes to fail
            # when the a large index is locking the database.
            sentence_table.create_index(['story_id'])
            sentence_table.create_index(['start_span'])
            sentence_table.create_index(['end_span'])

            db.query("PRAGMA journal_mode=WAL;")

        create_story_tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            async for lines, story_nums in chunk_stories_from_file(file_path, batch_size=batch_size):

                story_ids = [id for id in
                             list(db['story'].insert(dict(story_num=story_num)) for story_num in story_nums)]

                if marked_sentences:
                    sentence_splitter = MarkerSentenceSplitter()

                create_story_tasks.append(
                    loop.run_in_executor(executor, ProcessStory(sentence_splitter),
                                         lines, story_ids))

            for i, t in enumerate(asyncio.as_completed(create_story_tasks)):
                story_ids, sentences_to_save, story_metrics = await t

                db["sentence"].insert_many(sentences_to_save)

                for m in story_metrics:
                    db["story"].update(m, ['id'])

                print(f"Batch {i} - stories text saved: {story_ids}")

            logger.info(f"Saved stories to db with ids: {story_ids}")

            await save_language_features(batch_size, dataset_db, executor, loop)

            if should_save_sentiment:
                await save_sentiment(batch_size, dataset_db, executor, loop)

        if ner_model:
            await save_ner(ner_model, batch_size, dataset_db, cuda_device=cuda_device)

        if coreference_model:
            await save_coreferences(coreference_model, dataset_db, cuda_device=cuda_device)

    return dataset_db


async def save_sentiment(batch_size, dataset_db, executor, loop):
    tasks = []

    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
        sentence_sentiment_table = db.create_table('sentence_sentiment')
        sentence_sentiment_table.create_column('sentence_id', db.types.bigint)
        sentence_sentiment_table.create_column('vader_sentiment', db.types.float)
        sentence_sentiment_table.create_column('textblob_polarity', db.types.float)
        sentence_sentiment_table.create_column('textblob_subjectivity', db.types.float)
        sentence_sentiment_table.create_index(['sentence_id'])
        sentiment_batch = []

        for sentence in db['sentence']:
            sentiment_batch.append(sentence)

            if len(sentiment_batch) == batch_size:
                tasks.append(
                    loop.run_in_executor(executor, sentiment_features, sentiment_batch))
                sentiment_batch = []
        tasks.append(
            loop.run_in_executor(executor, sentiment_features, sentiment_batch))

        for i, t in enumerate(asyncio.as_completed(tasks)):
            result = await t
            db["sentence_sentiment"].insert_many(result)
            print(f"Sentence Sentiment batch saved {i}")

        logger.info(f"Sentiment saved")


async def save_language_features(batch_size, dataset_db, executor, loop):
    tasks = []

    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
        table = db.create_table('sentence_lang')
        table.create_column('sentence_id', db.types.bigint)
        table.create_column('lang', db.types.string)
        table.create_column('nonsense', db.types.boolean)
        table.create_column('ascii_chars', db.types.boolean)

        table.create_index(['sentence_id'])

        batch = []

        for sentence in db['sentence']:

            batch.append(sentence)

            if len(batch) == batch_size:
                tasks.append(loop.run_in_executor(executor, lang_features, batch))
                batch = []
        tasks.append(
            loop.run_in_executor(executor, lang_features, batch))

        for i, t in enumerate(asyncio.as_completed(tasks)):
            result = await t
            db["sentence_lang"].insert_many(result)
            print(f"Sentence Language batch saved {i}")

        logger.info(f"Language Features Saved")


async def save_ner(ner_model: Model, batch_size: int, dataset_db: str, cuda_device: Union[List[int], int] = None,
                   save_batch_size: int = 25):
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

        ner_batch = []

        gpu_max_workers = 1

        if isinstance(cuda_device, (list, tuple)):
            gpu_max_workers = len(cuda_device)
            gpus = cuda_device
        else:
            gpus = [cuda_device]

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=gpu_max_workers) as executor:

            processors = []
            for gpu in gpus:
                processors.append(NERProcessor(ner_model, dataset_db, cuda_device=gpu))
            processors_cycle = itertools.cycle(processors)

            tasks = []

            for sentence in db['sentence']:
                ner_batch.append(sentence)

                if len(ner_batch) == batch_size:
                    tasks.append(loop.run_in_executor(executor, next(processors_cycle), ner_batch))

                    if len(tasks) == save_batch_size:
                        results = await asyncio.gather(*tasks)
                        for res in results:
                            db["named_entity_sentence"].insert_many(res)
                        tasks = []
                    ner_batch = []

            tasks.append(
                loop.run_in_executor(executor, next(processors_cycle),
                                     ner_batch))
            results = await asyncio.gather(*tasks)
            for res in results:
                db["named_entity_sentence"].insert_many(res)

            db["named_entity_sentence"].create_index(['sentence_id'])

            logger.info(f"Named Entity Tags Saved")


async def save_coreferences(coreference_model: Model, dataset_db: str, cuda_device: Union[List[int], int] = None,
                            save_batch_size: int = 25, sentence_chunks: int = 200):
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

        coref_table = db.create_table('coreference')
        coref_table.create_column('story_id', db.types.bigint)
        coref_table.create_column('coref_id', db.types.integer)
        coref_table.create_column('start_span', db.types.integer)
        coref_table.create_column('end_span', db.types.integer)
        coref_table.create_column('mention_text', db.types.string)
        coref_table.create_column('context_text', db.types.string)
        coref_table.create_index(['story_id'])
        coref_table.create_index(['start_span'])
        coref_table.create_index(['end_span'])

        gpu_max_workers = 1

        if isinstance(cuda_device, (list, tuple)):
            gpu_max_workers = len(cuda_device)
            gpus = cuda_device
        else:
            gpus = [cuda_device]

        word_tokenizer = WordTokenizer()

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=gpu_max_workers) as executor:

            processors = []
            for gpu in gpus:
                processors.append(CoreferenceProcessor(coreference_model, dataset_db, cuda_device=gpu))
            processors_cycle = itertools.cycle(processors)

            tasks = []
            # Order by shortest to longest so possible failures are at the end.
            for story in db['story'].find(order_by=['sentence_num', 'id']):

                sentence_list = [s["text"] for s in db["sentence"].find(story_id=story["id"], order_by='id')]
                sentence_tokens = word_tokenizer.batch_tokenize(sentence_list)

                for sentence_chunk in more_itertools.chunked(sentence_tokens, n=sentence_chunks):
                    sentence_chunk_flat = list(more_itertools.flatten(sentence_chunk))

                    if len(sentence_chunk_flat) < 10:
                        continue

                    sentence_chunk_text = [t.text for t in sentence_chunk_flat]

                    tasks.append(loop.run_in_executor(executor, next(processors_cycle), sentence_chunk_text, story["id"]))

                    if len(tasks) == save_batch_size:
                        results = await asyncio.gather(*tasks)

                        for coref_to_save in results:
                            try:

                                db["coreference"].insert_many(copy.deepcopy(coref_to_save))

                            except Exception as e:
                                logging.error(e)

                        tasks = []

            results = await asyncio.gather(*tasks)

            for coref_to_save in results:
                try:

                    db["coreference"].insert_many(coref_to_save)

                except Exception as e:
                    logging.error(e)

            logger.info(f"Coreferences Saved")

def update_table_on_id(db, table, data):
    try:
        sentence_table = db[table]
        for sent_dict in data:
            sentence_table.update(sent_dict, ["id"])
    except Exception as e:
        logging.error(e)

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
            line_count += 1
            if len(lines) == batch_size:
                yield lines, story_nums
                lines = []
                story_nums = []

    yield lines, story_nums


def sentiment_features(story_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sents_to_save = []
    for sent_dict in story_sentences:
        analyzer = SentimentIntensityAnalyzer()
        text = sent_dict["text"]

        vader_sentiment = analyzer.polarity_scores(text)
        vader_compound = vader_sentiment["compound"]

        text_blob = TextBlob(text)
        polarity = text_blob.sentiment.polarity
        subjectivity = text_blob.sentiment.subjectivity

        sentiment_dict = dict(sentence_id=sent_dict["id"], vader_sentiment=vader_compound, textblob_polarity=polarity,
                              textblob_subjectivity=subjectivity)

        sents_to_save.append(sentiment_dict)

    return sents_to_save


def lang_features(story_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lang_list = []

    wtl = WhatTheLang()

    for sent_dict in story_sentences:

        text = sent_dict["text"]

        try:
            lang = wtl.predict_lang(text)

            if not isinstance(lang, str):
                lang = "UKN"

        except:
            lang = "UKN"
        try:
            if len(text) <= 10:
                is_nonsense = False
            else:
                is_nonsense = nonsense(text)
        except:
            is_nonsense = True

        is_eng = isAscii(text)

        lang_dict = dict(sentence_id=sent_dict["id"], lang=lang, nonsense=is_nonsense, ascii_chars=is_eng)

        lang_list.append(lang_dict)

    return lang_list


class MarkerSentenceSplitter():
    """ Split sentences based on a predefined marker.
    """

    def __init__(self, start_marker="[STR_SENT]", end_marker="[END_SENT]"):
        self.start_marker = start_marker
        self.end_marker = end_marker

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits a ``text`` :class:`str` paragraph into a list of :class:`str`, where each is a sentence.
        """
        split_text = text.split(self.start_marker)
        split_text = [t.replace(self.end_marker, "").replace(self.start_marker, "").strip() for t in split_text]
        split_text = [t for t in split_text if len(t) > 0]
        return split_text

    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        """
        Default implementation is to just iterate over the texts and call ``split_sentences``.
        """
        return [self.split_sentences(text) for text in texts]


class ProcessStory:
    def __init__(self, sentence_splitter):
        self._sentence_splitter = sentence_splitter
        self._word_tokenizer = WordTokenizer()

    def __call__(self, lines: List[List[str]], story_ids: List[int]) -> Tuple:

        story_metrics = []
        story_sentences_to_save = []

        story_sentences_split = self._sentence_splitter.batch_split_sentences(lines)

        for sentences, story_id in zip(story_sentences_split, story_ids):
            try:

                tokenized_sentences = self._word_tokenizer.batch_tokenize(sentences)

                total_story_tokens = 0

                for i, sentence in enumerate(tokenized_sentences):
                    start_span = total_story_tokens
                    sentence_len = len(sentence)
                    total_story_tokens += sentence_len
                    end_span = total_story_tokens

                    text = " ".join([s.text for s in sentence])

                    story_sentences_to_save.append(
                        dict(sentence_num=i, story_id=story_id, text=text,
                             sentence_len=sentence_len, start_span=start_span, end_span=end_span))

                story_metrics.append(dict(sentence_num=len(sentences), tokens_num=total_story_tokens, id=story_id))

            except Exception as e:
                logging.error(e)

        return story_ids, story_sentences_to_save, story_metrics


def negative_sentence_sampler(db: Database) -> Dict[str, Any]:
    while True:
        random_sentences = db.query(f'SELECT * FROM sentence ORDER BY RANDOM()')
        for sentence in random_sentences:
            yield sentence


class NERProcessor(object):
    def __init__(self, ner_model: str, database_db: str, cuda_device: Union[List[int], int] = -1):
        self._ner_predictor = Predictor.from_path(ner_model)

        if cuda_device != -1:
            self._ner_predictor._model = self._ner_predictor._model.to(cuda_device)
        self._database_db = database_db
        self._cuda_device = cuda_device

    def __call__(self, stories_sentences: List[Dict[str, Any]]) -> List[List[str]]:

        stories_ner = []

        try:
            batch_json = [{"sentence": story["text"], "cuda_device": self._cuda_device} for story in stories_sentences]

            batch_result = self._ner_predictor.predict_batch_json(
                batch_json
            )
            for ner_dict, sentence_dict in zip(batch_result, stories_sentences):
                ner_tags = dict(sentence_id=sentence_dict["id"], story_id=sentence_dict["story_id"],
                                ner_tags=" ".join(ner_dict["tags"]))
                stories_ner.append(ner_tags)
        except Exception as e:
            print(e)


        return stories_ner


class CoreferenceProcessor(object):
    def __init__(self, coreference_model: str, database_db: str, cuda_device: Union[List[int], int] = -1):
        self._coreference_predictor = Predictor.from_path(coreference_model)

        if cuda_device != -1:
            self._coreference_predictor._model = self._coreference_predictor._model.to(cuda_device)

        self._database_db = database_db
        self._cuda_device = cuda_device

    def __call__(self, stories_sentences: List[str], story_id: int) -> List[List[str]]:
        coreference_clusters = []

        try:
            result = self._coreference_predictor.predict_tokenized(
                tokenized_document=stories_sentences
            )

            clusters = result["clusters"]
            for i, cluster in enumerate(clusters):
                for span in cluster:
                    mention_text = " ".join(stories_sentences[span[0] : min(span[1] + 1,len(stories_sentences) - 1)])
                    context_text = " ".join(stories_sentences[max(0, span[0] - 5): min(span[1] + 5,len(stories_sentences) - 1)])
                    coreference_clusters.append(dict(coref_id=i, story_id=story_id, start_span=span[0], end_span=span[1], mention_text=mention_text, context_text=context_text))
        except Exception as e:
            print(e)
        return coreference_clusters
