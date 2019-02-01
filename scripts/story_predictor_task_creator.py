import argparse
import random
from collections import defaultdict, namedtuple, OrderedDict

import jsonlines
from allennlp.data.tokenizers import WordTokenizer

BucketTuple = namedtuple('Bucket', ['lower_bound', 'upper_bound'])


def main(args):
    print(f"Arguments: {args}")

    attribute_to_use = args["attribute_to_use"]

    word_tokenizer = WordTokenizer()

    buckets = []

    bucket_strings = [i.split('-') for i in args['buckets']]

    for lower, upper in bucket_strings:
        buckets.append(BucketTuple(lower_bound=float(lower), upper_bound=float(upper)))

    buckets_map = OrderedDict({i: v for i, v in enumerate(buckets)})

    story_buckets_map = defaultdict(lambda: defaultdict(lambda: list()))
    story_text_map = defaultdict(lambda: defaultdict(lambda: list()))

    with jsonlines.open(args["source_json"], mode='r') as reader:
        for json_obj in reader:

            story_id = json_obj["metadata"]["story_id"]

            source_text = json_obj["metadata"]["source_text"]
            target_text = json_obj["metadata"]["target_text"]
            source_len = len(word_tokenizer.tokenize(source_text))
            target_len = len(word_tokenizer.tokenize(target_text))

            story_text_map[story_id]["source_text"] = source_text
            story_text_map[story_id]["target_text"] = target_text

            attribute = float(json_obj[attribute_to_use])
            # TODO: Restrict to in length.
            for i, bucket in buckets_map.items():
                if attribute >= bucket.lower_bound and attribute < bucket.upper_bound:

                    if source_len < args["min_word_length"] or target_len < args["min_word_length"]:
                        continue

                    story_buckets_map[story_id][i].append(json_obj)

    with jsonlines.open(args["target_json"], mode='w') as writer:
        for story_id, buckets in story_buckets_map.items():
            # If at least one from each of the buckets is in the story then randomly select one.
            if all([len(buckets[b]) > 0 for b in buckets_map.keys()]):
                selection = []
                for i, contexts in buckets.items():
                    selected = random.choice(contexts)
                    selected["bucket"] = i
                    selection.append(selected)

                random.shuffle(selection)
                task_map = {"story_id": story_id, "all_story_text": story_text_map[story_id], "selection": selection}
                print(task_map)
                writer.write(task_map)


parser = argparse.ArgumentParser(
    description='Reads an Story Predictor output and adds buckets into the precentile output')
parser.add_argument('--source-json', required=True, type=str, help="The source JSON lines file.")
parser.add_argument('--target-json', required=True, type=str, help="The target JSON lines file.")
parser.add_argument('--attribute-to-use', type=str, default="neighbour_correct_score_percentile",
                    help="Which attribute to use to bucket the task.")
parser.add_argument('--buckets',
                    default=["0.0-0.25", "0.25-0.50", "0.5-0.75", "0.75-1.01"], type=float, nargs='+',
                    help="A list percentile pairs. Do as pairs as may not want to select every bucket. Define from largest to smallest.")
parser.add_argument('--min-word-length', default=5, type=int, help="")

args = parser.parse_args()
main(vars(args))
