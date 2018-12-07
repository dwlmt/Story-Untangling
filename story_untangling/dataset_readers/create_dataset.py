import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, List, Dict, Any

import dataset
from aiofile import AIOFile, LineReader
from allennlp.data.tokenizers import SentenceSplitter
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.predictors import Predictor
from dataset import Database
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import nltk

nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


async def create_dataset_db(dataset_path: str, db_discriminator: str, file_path: str, use_existing_database=True,
                            sentence_splitter: SentenceSplitter = SpacySentenceSplitter(), save_sentiment: bool = True,
                            batch_size: int = 100,
                            max_workers: int = 16) -> str:
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
        create_story_tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            async for lines, story_nums in chunk_stories_from_file(file_path, batch_size=batch_size):
                story_sentences = sentence_splitter.batch_split_sentences(lines)
                create_story_tasks.append(
                    loop.run_in_executor(executor, SaveStoryToDatabase(dataset_db), story_sentences, story_nums))

            story_ids = await asyncio.gather(*create_story_tasks)
            logger.info(f"Saved stories to db with ids: {story_ids}")

            if save_sentiment:
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

            try:
                db["sentence"].create_index(['story_id'])
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


class SentimentDatabaseFeatures:
    def __init__(self, dataset_db):
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

        db = dataset.connect(self._dataset_db, engine_kwargs={"pool_recycle": 3600})
        try:
            db.begin()
            sentence_table = db["sentence"]
            for sent_dict in sents_to_save:
                sentence_table.update(sent_dict, ["id"])
            db.commit()
        except:
            db.rollback()


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
                    start_token = total_story_tokens
                    sentence_len = len(sent)
                    total_story_tokens += sentence_len
                    end_token = total_story_tokens
                    sentences_to_save.append(
                        dict(sentence_num=i, story_id=story_id, text=sent, sentence_len=sentence_len,
                             start_token=start_token, end_token=end_token))
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


class NERProcessor(object):
    def __init__(self, ner_model: str):
        self._ner_predictor = Predictor.from_path(ner_model)

    def __call__(self, stories_sentences: List[List[str]]) -> List[List[str]]:
        stories_ner = []
        for story_sentences in stories_sentences:
            batch_json = [{"sentence": story} for story in story_sentences]

            batch_result = self._ner_predictor.predict_batch_json(
                batch_json
            )

            tags = [" ".join(ner_dict["tags"]) for ner_dict in batch_result]
            stories_ner.append(tags)
        return stories_ner
