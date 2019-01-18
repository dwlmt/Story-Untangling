import asyncio
import copy
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

import dataset
import more_itertools
import nltk
from aiofile import AIOFile, LineReader
from allennlp.data.tokenizers import SentenceSplitter, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from dataset import Database
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}


async def create_dataset_db(dataset_path: str, db_discriminator: str, file_path: str, use_existing_database=True,
                            sentence_splitter: SentenceSplitter = SpacySentenceSplitter(),
                            should_save_sentiment: bool = True,
                            ner_model: str = None,
                            coreference_model: str = None,
                            batch_size: int = 100,
                            max_workers: int = 16,
                            truncate_sequence_length : int = 250,
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

        f"{dataset_db}?mode=ro&cache=shared"
        with dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600, "connect_args": {'timeout': 300}}) as db:

            
            # Create the main tables and columns that need indexing.
            story_table = db.create_table('story')
            story_table.create_column('story_num', db.types.integer)
            sentence_table = db.create_table('sentence')
            sentence_table.create_column('story_id', db.types.integer)
            sentence_table.create_column('sentence_num', db.types.integer)
            sentence_table.create_column('sentence_len', db.types.integer)
            sentence_table.create_column('start_span', db.types.integer)
            sentence_table.create_column('end_span', db.types.integer)
            # Indices created at the beginning as creating them later causing other processes to fail
            # when the a large index is locking the database.
            sentence_table.create_index(['story_id'])
            sentence_table.create_index(['start_span'])
            sentence_table.create_index(['end_span'])

        create_story_tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            async for lines, story_nums in chunk_stories_from_file(file_path, batch_size=batch_size):
                story_sentences = sentence_splitter.batch_split_sentences(lines)
                create_story_tasks.append(
                    loop.run_in_executor(executor, SaveStoryToDatabase(dataset_db, truncate_sequence_length=truncate_sequence_length),
                                         story_sentences, story_nums))

            story_ids = await asyncio.gather(*create_story_tasks)
            logger.info(f"Saved stories to db with ids: {story_ids}")

            if should_save_sentiment:
                await save_sentiment(batch_size, dataset_db, executor, loop)

        if ner_model:
            await save_ner(ner_model, batch_size, dataset_db, cuda_device=cuda_device)

        if coreference_model:
            await save_coreferences(coreference_model, dataset_db, cuda_device=cuda_device)

    return dataset_db


async def save_sentiment(batch_size, dataset_db, executor, loop):
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
        db["sentence"].create_column('vader_sentiment', db.types.float)
        db["sentence"].create_column('textblob_polarity', db.types.float)
        db["sentence"].create_column('textblob_subjectivity', db.types.float)
        sentiment_batch = []
        sentiment_tasks = []
        for sentence in db['sentence']:
            sentiment_batch.append(sentence)

            if len(sentiment_batch) == batch_size:
                sentiment_tasks.append(
                    loop.run_in_executor(executor, SentimentDatabaseFeatures(dataset_db), sentiment_batch))
                sentiment_batch = []
        sentiment_tasks.append(
            loop.run_in_executor(executor, SentimentDatabaseFeatures(dataset_db), sentiment_batch))
        await asyncio.gather(*sentiment_tasks)
        logger.info(f"Sentiment saved")


async def save_ner(ner_model: Model, batch_size: int, dataset_db: str, cuda_device: Union[List[int], int] = None,
                   save_batch_size: int = 50):
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
        #db.begin()
        db["sentence"].create_column('ner_tags', db.types.text)
        #db.commit()

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
                    for ner_data_to_save in results:
                        update_table_on_id(db, "sentence", ner_data_to_save)
                    tasks = []
                ner_batch = []

        tasks.append(
            loop.run_in_executor(executor, next(processors_cycle),
                                 ner_batch))
        results = await asyncio.gather(*tasks)
        for ner_data_to_save in results:
            with dataset.connect(dataset_db,
                                 engine_kwargs=engine_kwargs) as db:
                update_table_on_id(db, "sentence", ner_data_to_save)

        logger.info(f"Named Entity Tags Saved")


async def save_coreferences(coreference_model: Model, dataset_db: str, cuda_device: Union[List[int], int] = None,
                            save_batch_size: int = 50, sentence_chunks: int = 100):
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

        #db.begin()
        coref_table = db.create_table('coreference')
        coref_table.create_column('story_id', db.types.integer)
        coref_table.create_column('start_span', db.types.integer)
        coref_table.create_column('end_span', db.types.integer)
        coref_table.create_index(['story_id'])
        coref_table.create_index(['start_span'])
        coref_table.create_index(['end_span'])
        #db.commit()

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
            processors.append(CoreferenceProcessor(coreference_model, dataset_db, cuda_device=gpu))
        processors_cycle = itertools.cycle(processors)

        tasks = []
        # Order by shortest to longest so possible failures are at the end.
        for story in db['story'].find(order_by=['sentence_num','id']):

            sentence_list = [s["text"] for s in db["sentence"].find(story_id=story["id"], order_by='id')]

            for sentence_chunk in more_itertools.chunked(sentence_list, n=sentence_chunks):
                sentence_text = " ".join(sentence_chunk)

                tasks.append(loop.run_in_executor(executor, next(processors_cycle), sentence_text, story["id"]))

                if len(tasks) == save_batch_size:
                    results = await asyncio.gather(*tasks)


                    for coref_to_save in results:
                        with dataset.connect(dataset_db,
                                             engine_kwargs=engine_kwargs) as db:
                            try:

                                #db.begin()
                                db["coreference"].insert_many(copy.deepcopy(coref_to_save))
                                #db.commit()
                            except Exception as e:
                                logging.error(e)
                                #db.rollback()
                    tasks = []

        results = await asyncio.gather(*tasks)


        for coref_to_save in results:
            with dataset.connect(dataset_db,
                                 engine_kwargs=engine_kwargs) as db:
                try:
                    #db.begin()
                    db["coreference"].insert_many(coref_to_save)
                    #db.commit()
                except Exception as e:
                    logging.error(e)
                    #db.rollback()

        logger.info(f"Coreferences Saved")


def update_table_on_id(db, table, data):
    try:
        #db.begin()
        sentence_table = db[table]
        for sent_dict in data:
            sentence_table.update(sent_dict, ["id"])
        #db.commit()

    except Exception as e:
        logging.error(e)
        #db.rollback()


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


class SentimentDatabaseFeatures:
    def __init__(self, dataset_db: str):
        self._dataset_db = dataset_db

    def __call__(self, story_sentences: List[Dict[str, Any]]) -> None:
        sents_to_save = []
        for sent_dict in story_sentences:
            analyzer = SentimentIntensityAnalyzer()
            text = sent_dict["text"]

            vader_sentiment = analyzer.polarity_scores(text)
            vader_compound = vader_sentiment["compound"]

            text_blob = TextBlob(text)
            polarity = text_blob.sentiment.polarity
            subjectivity = text_blob.sentiment.subjectivity

            sentiment_dict = dict(id=sent_dict["id"], vader_sentiment=vader_compound, textblob_polarity=polarity,
                                  textblob_subjectivity=subjectivity)

            sents_to_save.append(sentiment_dict)

        with dataset.connect(self._dataset_db,
                             engine_kwargs=engine_kwargs) as db:
            update_table_on_id(db, "sentence", sents_to_save)


class SaveStoryToDatabase:
    def __init__(self, dataset_db: str, truncate_sequence_length: int = 250):
        self._dataset_db = dataset_db
        self._truncate_sequence_length = truncate_sequence_length
        self._word_tokenizer = WordTokenizer()

    def __call__(self, story_sentences: List[str], story_nums: List[int]) -> List[int]:
        story_ids = []
        for sentences, story_num in zip(story_sentences, story_nums):
            with dataset.connect(self._dataset_db,
                                 engine_kwargs=engine_kwargs) as db:

                try:
                    #db.begin()
                    story_table = db['story']
                    sentence_table = db['sentence']
                    story = dict(story_num=story_num)
                    story_id = story_table.insert(story)
                    sentences_to_save = []

                    total_story_tokens = 0
                    sentences = self._word_tokenizer.batch_tokenize(sentences)

                    for i, sent in enumerate(sentences):
                        if self._truncate_sequence_length:
                            sent = sent[0:min(self._truncate_sequence_length, len(sent))]
                        start_span = total_story_tokens
                        sentence_len = len(sent)
                        total_story_tokens += sentence_len
                        end_span = total_story_tokens

                        text = " ".join([s.text for s in sent])

                        sentences_to_save.append(
                            dict(sentence_num=i, story_id=story_id, text=text,
                                 sentence_len=sentence_len,
                                 start_span=start_span, end_span=end_span))
                    sentence_table.insert_many(sentences_to_save)

                    story_table.update(dict(sentence_num=len(sentences), tokens_num=total_story_tokens, id=story_id),
                                       ['id'])
                    story_ids.append(story_id)
                    #db.commit()

                except Exception as e:
                    logging.error(e)
                    #db.rollback()
                    
        return story_ids


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

        batch_json = [{"sentence": story["text"], "cuda_device": self._cuda_device} for story in stories_sentences]

        batch_result = self._ner_predictor.predict_batch_json(
            batch_json
        )
        for ner_dict, sentence_dict in zip(batch_result, stories_sentences):
            ner_tags = dict(id=sentence_dict["id"], ner_tags=" ".join(ner_dict["tags"]))
            stories_ner.append(ner_tags)
        return stories_ner


class CoreferenceProcessor(object):
    def __init__(self, corefernce_model: str, database_db: str, cuda_device: Union[List[int], int] = -1):
        self._coreference_predictor = Predictor.from_path(corefernce_model)

        if cuda_device != -1:
            self._coreference_predictor._model = self._coreference_predictor._model.to(cuda_device)

        self._database_db = database_db
        self._cuda_device = cuda_device

    def __call__(self, stories_sentences: str, story_id: int) -> List[List[str]]:

        result = self._coreference_predictor.predict(
            document=stories_sentences
        )
        coreference_clusters = []
        clusters = result["clusters"]
        for i, cluster in enumerate(clusters):
            for span in cluster:
                coreference_clusters.append(dict(coref_id=i, story_id=story_id, start_span=span[0], end_span=span[1]))

        return coreference_clusters