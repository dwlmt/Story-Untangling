import argparse
import csv

import dataset

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}


def sample_stories(args):
    print(args)

    database = args["database"]
    dataset_db = f"sqlite:///{database}"

    csv_lines = []

    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

        stories = db.query(
            f'SELECT * FROM story  WHERE sentence_num >= {args["min_sentences"]} '
            f'AND sentence_num <= {args["max_sentences"]} ORDER BY RANDOM() LIMIT {args["num_samples"]}')

        for story in stories:
            sentences = db.query(
                f'SELECT * FROM sentence WHERE story_id = {story["id"]} ORDER BY id ')
            sentence_text = [s["text"].replace("<newline>", " ") for s in sentences]

            story_text = " ".join(sentence_text)
            csv_lines.append((int(story["id"]), story_text))

    with open(args["target"], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["story_id", "text"])
        for line in csv_lines:
            writer.writerow(line)


# TODO: Setup initiall for

parser = argparse.ArgumentParser(
    description='Randomly sample from the database stories')
parser.add_argument('--database', required=True, type=str, help="Output the saved weights of the Topic Model")
parser.add_argument('--target', required=True, type=str, help="Output CSV file")
parser.add_argument('--min-sentences', type=int, default=25, help="The min number of sentences.")
parser.add_argument('--max-sentences', type=int, default=75, help="The max number of sentences.")
parser.add_argument('--num-samples', type=int, default=500,
                    help="Number of stories to sample from. Should be less than the size of the dataset.")

args = parser.parse_args()

sample_stories(vars(args))
