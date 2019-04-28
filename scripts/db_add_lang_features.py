import argparse
import asyncio
from concurrent.futures.process import ProcessPoolExecutor

from story_untangling.dataset_readers.dataset_features import save_language_features

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}


async def add_lang_features(args):
    database = args["database"]
    dataset_db = f"sqlite:///{database}"

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=args["max_workers"]) as executor:
        await save_language_features(args["batch_size"], dataset_db, executor, loop)


parser = argparse.ArgumentParser(
    description='Add per sentence sentiment information to the database.')
parser.add_argument('--database', required=True, type=str, help="Output the saved weights of the Topic Model")
parser.add_argument('--batch-size', type=int, default=100, help="Size of the batch to process.")
parser.add_argument('--max-workers', type=int, default=12, help="Number of topics to use from HDP. Default: 12")

args = parser.parse_args()

loop = asyncio.get_event_loop()
dataset_db = loop.run_until_complete(add_lang_features(vars(args)))
