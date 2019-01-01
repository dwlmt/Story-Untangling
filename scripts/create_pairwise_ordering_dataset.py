import argparse
import random

import dataset
import jsonlines


def main(args):
    print(f"Create dataset with args: {args}")

    dataset_db = args["source_db"]
    db_name = f"sqlite:///{dataset_db}"
    db = dataset.connect(db_name, engine_kwargs={"pool_recycle": 3600})

    with jsonlines.open(args["target_json"], mode='w') as writer:

        stories = db.query(
            f'SELECT * FROM story  WHERE sentence_num >= {args["min_story_sentences"]} '
            f'AND sentence_num <= {args["max_story_sentences"]} ORDER BY id')

        for story in stories:

            out_story_sentences = []
            # Add the database link to be able to reuse more complicated presaved features such as NER and coreferences.
            # Needs to be per story as the in the AllenNLP predictor API each json lines has self contained parameters.
            story["db"] = args["source_db"]
            story["sentences"] = out_story_sentences

            sentences = [s for s in
                         db.query(f'SELECT * FROM sentence WHERE story_id = {story["id"]} ORDER BY sentence_num')]

            # Randomly select a bloc of contiguous sentences from the list.
            N = len(sentences)
            L = args["pairwise_sequence_length"]

            start = random.randint(0, N - L)
            sentences = sentences[start:start + L]

            # Shuffle all sentences apart from the original so the first is always in the correct order.
            sentences_copy = sentences[2:]
            random.shuffle(sentences_copy)
            sentences[2:] = sentences_copy

            for sentence in sentences:
                out_story_sentences.append(sentence)

            writer.write(story)


parser = argparse.ArgumentParser(description='Create a dataset for running a pairwise predictor.')
parser.add_argument('--source-db', required=True, type=str, help="The source SQLITE dataset file.")
parser.add_argument('--target-json', default="./pairwise_predictor.jsonl", type=str,
                    help="The target JSON lines for the predictor file. Default: ./pairwise_preidctor.jsonl")
parser.add_argument('--pairwise-sequence-length', default=8, type=int,
                    help="Randomly sample a block of this size from the story. Default: 8")
parser.add_argument('--max-story-sentences', default=500, type=int,
                    help="Max number of sentences a story must have to be included. Default: 500")
parser.add_argument('--min-story-sentences', default=8, type=int,
                    help="Min number of sentences a story must have to be included. Default: 8")

args = parser.parse_args()
main(vars(args))
