import argparse
import asyncio

import dataset
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from story_untangling.dataset_readers.dataset_features import update_table_on_id

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}


def add_sentiment(args):
    print(args)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(add_sentiment_to_db(args))


async def add_sentiment_to_db(args):
    database = args["database"]
    dataset_db = f"sqlite:///{database}"

    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
        db.begin()
        db["sentence"].create_column('vader_sentiment', db.types.float)
        db["sentence"].create_column('textblob_polarity', db.types.float)
        db["sentence"].create_column('textblob_subjectivity', db.types.float)
        db.commit()
        batch = []

        analyzer = SentimentIntensityAnalyzer()

        for sent_dict in db['sentence']:

            text = sent_dict["text"]

            vader_sentiment = analyzer.polarity_scores(text)
            vader_compound = vader_sentiment["compound"]

            text_blob = TextBlob(text)
            polarity = text_blob.sentiment.polarity
            subjectivity = text_blob.sentiment.subjectivity

            sentiment_dict = dict(id=sent_dict["id"], vader_sentiment=vader_compound, textblob_polarity=polarity,
                                  textblob_subjectivity=subjectivity)

            batch.append(sentiment_dict)

            if len(batch) == args["batch_size"]:
                update_table_on_id(db, "sentence", batch)
                batch = []

        if len(batch) > 0:
            update_table_on_id(db, "sentence", batch)


parser = argparse.ArgumentParser(
    description='Add per sentence sentiment information to the database.')
parser.add_argument('--database', required=True, type=str, help="Output the saved weights of the Topic Model")
parser.add_argument('--batch-size', type=int, default=100, help="Output the saved weights of the Topic Model")

args = parser.parse_args()

add_sentiment(vars(args))
