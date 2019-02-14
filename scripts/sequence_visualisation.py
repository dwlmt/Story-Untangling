import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jsonlines import jsonlines

sns.set(font_scale=0.50)

sns.set_style("whitegrid")


def main(args):
    print(args)

    results_dir = args["results_dir"]

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    source_stories = process_source_json(
        args)

    for story_id, story_data in source_stories.items():
        X_abs = [d[0] for d in story_data]
        X_rel = [d[1] for d in story_data]
        Y = [d[2] for d in story_data]

        data = pd.DataFrame.from_dict({"position": X_abs, "attribute": Y})
        plot_line(data=data, plot_name=f'{results_dir}/{story_id}_abs_pos_line.pdf')

        data = pd.DataFrame.from_dict({"position": X_rel, "attribute": Y})
        plot_line(data=data, plot_name=f'{results_dir}/{story_id}_rel_pos_line.pdf')


def plot_line(data, X="position", Y="attribute", plot_name=None):
    ax = sns.lineplot(data=data, x=X, y=Y)

    ymin, ymax = ax.get_ylim()
    bonus = (ymax - ymin) / 50
    for x, y in zip(data["position"], data["attribute"]):
        ax.text(x, y - bonus, str(x), color='gray')

    fig = ax.get_figure()
    fig.savefig(plot_name)

    fig.clear()
    plt.close(fig)


def process_source_json(args):
    # Map story id and position to dicts so can be extracted and analysed separately.
    story_map = defaultdict(lambda: list())

    with jsonlines.open(args["source_json"], mode='r') as reader:
        for i, json_obj in enumerate(reader):
            story_id = json_obj["metadata"]["story_id"]
            abs_pos = json_obj["metadata"]["absolute_position"]
            rel_pos = json_obj["metadata"]["relative_position"]

            attribute = json_obj[args["attribute"]]

            story_map[story_id].append((abs_pos, rel_pos, attribute))

    return story_map


parser = argparse.ArgumentParser(
    description='Performs PCA and creates a T-SNE visualisation')
parser.add_argument('--source-json', required=True, type=str, help="The source data csv.")
parser.add_argument('--results-dir', default="./timeseries_visualisation/", type=str, help="The source data csv.")
parser.add_argument('--attribute', required=True, type=str, help="The attribute to use when creating the charts.")

args = parser.parse_args()

main(vars(args))
