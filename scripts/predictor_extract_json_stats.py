import argparse
import os

import jsonlines
import pandas
from tqdm import tqdm


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


parser = argparse.ArgumentParser(
    description='Extract JSON perform dimensionality reduction and save window, position and batch information.')
parser.add_argument('--source-json', required=True, type=str, help="JSON file to process.")
parser.add_argument('--output-dir', required=True, type=str,
                    help="Output for the Panda data frames with the extracted stats.")

args = parser.parse_args()


def extract_json_stats(args):
    print(args)

    story_metadata = []
    position_stats = []
    batch_stats = []
    window_stats = []

    ensure_dir(args["output_dir"])

    source_basename = os.path.basename(args['source_json'])
    source_basename = source_basename.replace('.jsonl', '')

    with jsonlines.open(args['source_json']) as reader:
        for i, obj in tqdm(enumerate(reader)):

            if len(obj) == 0:
                continue

            children = obj["children"]
            for child in children:

                type = child["name"]

                if type == "position":
                    position_stats.extend(child["children"])

                    story_metadata.append({"story_id": child["children"][0]["story_id"]})

                elif type == "batch_stats":
                    for grandchild in child["children"]:
                        batch_stats.append(grandchild)
                elif type == "window_stats":

                    for window_variable in child["children"]:
                        window_size = window_variable["name"]

                        for window_slot in window_variable["children"]:

                            if "children" not in window_slot:
                                continue

                            window_stats.extend(
                                [{**w, **{"window_name": window_size, "window_size": window_slot["name"]}} for w in
                                 window_slot["children"]])

    print(f"Position {len(position_stats)}, Batch Stats: {len(batch_stats)}, Window Stats: {len(window_stats)}")

    position_df = pandas.DataFrame(data=position_stats)
    position_df.to_csv(f'{args["output_dir"]}/{source_basename}_position_stats.csv.xz')
    print(position_df)

    batch_stats_df = pandas.DataFrame(data=batch_stats)
    batch_stats_df.to_csv(f'{args["output_dir"]}/{source_basename}_batch_stats.csv.xz')
    print(batch_stats_df)

    window_stats_df = pandas.DataFrame(data=window_stats)
    window_stats_df.to_csv(f'{args["output_dir"]}/{source_basename}_window_stats.csv.xz')
    print(window_stats_df)


extract_json_stats(vars(args))
