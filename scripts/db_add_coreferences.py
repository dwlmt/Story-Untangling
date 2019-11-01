import argparse
import asyncio
from concurrent.futures.process import ProcessPoolExecutor

from story_untangling.dataset_readers.dataset_features import save_sentiment, save_ner, save_coreferences

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 1000, "check_same_thread": False}}


async def add_coreference_features(args):
    database = args["database"]
    dataset_db = f"sqlite:///{database}"

    await save_coreferences(args["model"], dataset_db, args["cuda_device"])

parser = argparse.ArgumentParser(
    description='Add coreferences to the database.')
parser.add_argument('--database', required=True, type=str, help="The database.")
parser.add_argument('--cuda-device', type=int, default=0, help="Cuda device: Default 0")
parser.add_argument('--model', type=str, default="https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz",
                    help="The Coreferences model to run.")

args = parser.parse_args()

loop = asyncio.get_event_loop()
dataset_db = loop.run_until_complete(add_coreference_features(vars(args)))