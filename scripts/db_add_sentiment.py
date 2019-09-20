import argparse
import asyncio
from concurrent.futures.process import ProcessPoolExecutor

from story_untangling.dataset_readers.dataset_features import save_sentiment

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 1000, "check_same_thread": False}}


async def add_sentiment_features(args):
    database = args["database"]
    dataset_db = f"sqlite:///{database}"

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=args["max_workers"]) as executor:
        await save_sentiment(args["batch_size"], dataset_db, executor, loop)


parser = argparse.ArgumentParser(
    description='Add per sentence sentiment information to the database.')
parser.add_argument('--database', required=True, type=str, help="The database.")
parser.add_argument('--batch-size', type=int, default=1000, help="Batch size. default: 1000")
parser.add_argument('--max-workers', type=int, default=16, help="Number of topics to use from HDP. Default: 16")

args = parser.parse_args()

add_sentiment_features(vars(args))
