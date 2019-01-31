import argparse
from collections import defaultdict

import jsonlines
import numpy as np
from scipy.stats import stats


def main(args):
    print(f"Arguments: {args}")

    attribute_values_to_rank = defaultdict(lambda: list())
    attribute_percentiles = defaultdict(lambda: list())

    with jsonlines.open(args["source_json"], mode='r') as reader:
        for json_obj in reader:

            for attr in args["attributes_to_bucket"]:
                if attr in json_obj:
                    attribute_values_to_rank[attr].append(json_obj[attr])

    for k, v in attribute_values_to_rank.items():
        attr_value_vec = np.array(v)
        # This corresponds to culmative distribution where x% have a values that is lower than or equal to this.
        attr_perc_rank = stats.rankdata(attr_value_vec, "max") / len(attr_value_vec)
        attribute_percentiles[k].extend(attr_perc_rank.tolist())

    attribute_keys_list = attribute_percentiles.keys()
    attribute_values_list = attribute_percentiles.values()
    attribute_percentiles_combined = []
    for values in zip(*attribute_values_list):
        percentiles_per_attr = {}
        for i, attr in enumerate(attribute_keys_list):
            percentiles_per_attr[f"{attr}_percentile"] = values[i]
        attribute_percentiles_combined.append(percentiles_per_attr)

    with jsonlines.open(args["source_json"], mode='r') as reader:
        with jsonlines.open(args["target_json"], mode='w') as writer:
            for json_obj, percentiles in zip(reader, attribute_percentiles_combined):
                out_json_obj = {**json_obj, **percentiles}
                print(out_json_obj)
                writer.write(out_json_obj)


parser = argparse.ArgumentParser(
    description='Reads an Story Predictor output and adds buckets into the precentile output')
parser.add_argument('--source-json', required=True, type=str, help="The source JSON lines file.")
parser.add_argument('--target-json', required=True, type=str, help="The target JSON lines file.")
parser.add_argument('--attributes_to_bucket',
                    default=["neighbour_correct_score", "neighbour_correct_score", "neighbour_correct_log_probs",
                             "neighbour_correct_similarity_cosine", "neighbour_correct_distance_l1",
                             "neighbour_correct_distance_l2"], type=str, nargs='+',
                    help="A list of attributes to bucket into the percentiles.")

args = parser.parse_args()
main(vars(args))
