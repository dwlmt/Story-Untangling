import argparse
import os
from collections import defaultdict

import dask.array as da
import hdbscan
import matplotlib.pyplot as plt
import numpy
import numpy as np
import seaborn as sns
from hdbscan import HDBSCAN
from jsonlines import jsonlines
from sklearn import preprocessing
from sklearn.manifold import TSNE
from umap import UMAP

sns.set(font_scale=0.50)

sns.set_style("whitegrid")


def main(args):
    print(args)

    results_dir = args["results_dir"]

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    source_embeddings_arr, target_embeddings_arr, story_ids, absolute_positions, story_ids_and_pos, story_id_map, positions_map = process_source_json(
        args)

    random_indices = numpy.random.choice(source_embeddings_arr.shape[0],
                                         size=min(args["vis_points"], source_embeddings_arr.shape[0]), )

    dim_red = UMAP(n_neighbors=args["umap_n_neighbours"],
                   n_components=args["dim_reduction_components"], metric=args["similarity_metric"])

    source_dim_red = da.from_array(dim_red.fit_transform(source_embeddings_arr), chunks=(1000, 1000))
    target_dim_red = da.from_array(dim_red.fit_transform(target_embeddings_arr), chunks=(1000, 1000))

    embeddings_to_process = {"source": source_dim_red, "target": target_dim_red}

    for embeddings_name, embeddings in embeddings_to_process.items():

        clusterer = hdbscan.HDBSCAN(algorithm='best', metric=args["clustering_metric"],
                                    min_cluster_size=args["min_cluster_size"])
        clusterer.fit(embeddings)

        cluster_labels = np.array(clusterer.labels_)
        cluster_probs = np.array(clusterer.probabilities_)
        cluster_outliers = np.array(clusterer.outlier_scores_)

        palette = sns.color_palette(palette="Spectral", n_colors=len(set(clusterer.labels_)))
        cluster_colors = np.array([sns.desaturate(palette[col], sat)
                                   if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                                   zip(clusterer.labels_, clusterer.probabilities_)])

        with jsonlines.open(f"{results_dir}/{embeddings_name}_cluster_membership.jsonl", mode='w') as writer:
            for story_id, pos, label, prob, outlier in zip(story_ids, absolute_positions, cluster_labels, cluster_probs,
                                                           cluster_outliers):
                writer.write({"story_id": int(story_id), "absolute_position": int(pos), "cluster": int(label),
                              "probability": float(prob), "outlier_score": outlier})

        sampled_embeddings = embeddings[random_indices]
        if not args["visualise_tsne"]:
            vis_data = UMAP(n_neighbors=args["umap_n_neighbours"],  # min_dist=args["umap_min_dist"],
                            n_components=args["umap_n_components"], metric=args["similarity_metric"]).fit_transform(
                sampled_embeddings)

        else:
            vis_data = TSNE().fit_transform(sampled_embeddings)

        X = vis_data[:, 0]
        Y = vis_data[:, 1]

        plot_scatter(X, Y, colors=cluster_colors[random_indices],
                     plot_name=f'{results_dir}/{embeddings_name}_scatter.pdf', size=2)

        ax = clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
        fig = ax.get_figure()
        fig.savefig(f'{results_dir}/{embeddings_name}_condensed_cluster_tree.pdf')
        fig.clear()
        plt.close(fig)

        for story_id, indices in story_id_map.items():
            story_embeddings = embeddings[indices]
            story_positions = absolute_positions[indices]
            story_cluster_colors = cluster_colors[indices]

            if not args["visualise_tsne"]:
                vis_data = UMAP(n_neighbors=args["umap_n_neighbours"],  # min_dist=["umap_min_dist"],
                                n_components=args["umap_n_components"], metric=args["similarity_metric"]).fit_transform(
                    story_embeddings)
            else:
                vis_data = TSNE().fit_transform(story_embeddings)

            X = vis_data[:, 0]
            Y = vis_data[:, 1]

            plot_scatter(X, Y, colors=story_cluster_colors,
                         plot_name=f'{results_dir}/{embeddings_name}_{story_id}_scatter.pdf', labels=story_positions)

            # If there are less datapoint than the dimensions then shrink the dimensions to allow clustering.
            if len(indices) <= args["dim_reduction_components"]:
                dim_components = max(min(len(indices), args["dim_reduction_components"]) - 2, 2)
                story_embeddings = UMAP(n_neighbors=args["umap_n_neighbours"],  # min_dist=args["umap_min_dist"],
                                        n_components=dim_components, metric=args["similarity_metric"]).fit_transform(
                    story_embeddings)

            story_clusterer = HDBSCAN(algorithm='best', metric=args["clustering_metric"],
                                      min_cluster_size=args["min_cluster_size"], approx_min_span_tree=False,
                                      gen_min_span_tree=True)
            story_clusterer.fit(story_embeddings)

            ax = story_clusterer.minimum_spanning_tree_.plot()

            d = ax.collections[0]

            d.set_offset_position('data')
            points = d.get_offsets()

            ymin, ymax = ax.get_ylim()
            color = "black"  # choose a color
            bonus = (ymax - ymin) / 50
            for point, name in zip(points, story_positions):
                x, y = point
                ax.text(x, y + bonus, name, color=color)

            fig = ax.get_figure()
            fig.savefig(f'{results_dir}/{embeddings_name}_{story_id}_mst.pdf')
            fig.clear()
            plt.close(fig)


def plot_scatter(X, Y, colors=None, plot_name=None, labels=None, size=None):
    fig, ax = plt.subplots()
    # sns.scatterplot(x=X, y=Y, legend=False, ax=ax, facecolors=colors)
    if size:
        ax.scatter(X, Y, color=colors, s=size)
    else:
        ax.scatter(X, Y, color=colors)
    ax.set(yticks=[])
    ax.set(xticks=[])

    if labels is not None:
        ymin, ymax = ax.get_ylim()
        bonus = (ymax - ymin) / 50
        for x, y, name, color in zip(X, Y, labels, colors):
            ax.text(x, y + bonus, name, color=color)
    fig = ax.get_figure()
    fig.savefig(plot_name)

    fig.clear()
    plt.close(fig)


def process_source_json(args):
    # Map story id and position to dicts so can be extracted and analysed separately.
    story_id_map = defaultdict(lambda: list())
    positions_map = defaultdict(lambda: list())
    story_ids = []
    absolute_positions = []
    story_ids_and_pos = []
    source_embeddings = []
    target_embeddings = []
    with jsonlines.open(args["source_json"], mode='r') as reader:
        for i, json_obj in enumerate(reader):
            story_id = json_obj["metadata"]["story_id"]
            abs_pos = json_obj["metadata"]["absolute_position"]

            story_id_map[story_id].append(i)
            positions_map[abs_pos].append(i)

            story_ids.append(story_id)
            absolute_positions.append(abs_pos)
            story_ids_and_pos.append(f"{story_id}_{abs_pos}")

            source_embeddings.append(json_obj["source_embeddings"])
            target_embeddings.append(json_obj["target_embeddings"])

    source_embeddings_arr = da.from_array(source_embeddings, chunks=(1000, 1000))
    target_embeddings_arr = da.from_array(target_embeddings, chunks=(1000, 1000))

    if args["normalize"]:
        source_embeddings_arr = da.from_array(preprocessing.scale(source_embeddings_arr), chunks=(1000, 1000))
        target_embeddings_arr = da.from_array(preprocessing.scale(target_embeddings_arr), chunks=(1000, 1000))

    return source_embeddings_arr, target_embeddings_arr, story_ids, np.array(
        absolute_positions), story_ids_and_pos, story_id_map, positions_map


parser = argparse.ArgumentParser(
    description='Performs PCA and creates a T-SNE visualisation')
parser.add_argument('--source-json', required=True, type=str, help="The source data csv.")
parser.add_argument('--results-dir', default="./embedding_visualisation/", type=str, help="The source data csv.")
parser.add_argument('--cache', default="./disk/scratch/s1569885", type=str, help="The cache for clustering")
parser.add_argument('--dim-reduction-components', default=50, type=int, help="The number of PCA components.")
parser.add_argument('--vis-points', default=20000, type=int, help="Max number of points for visualisation")
parser.add_argument('--umap-n-neighbours', default=25, type=int, help="The number of neighbours.")
parser.add_argument('--umap-min-dist', default=0.1, type=float, help="Controls how clumpy umap will cluster points.")
parser.add_argument('--umap-n-components', default=2, type=int, help="Number of components to reduce to.")
parser.add_argument('--similarity-metric', default="euclidean", type=str, help="The similarity measure to use.")
parser.add_argument('--clustering-metric', default="euclidean", type=str, help="The similarity measure to use.")
parser.add_argument('--min-cluster-size', default=5, type=int, help="Min size fo each cluster.")
parser.add_argument('--normalize', default=False, action='store_true', help="Normalize to mean of zero and std of 1")
parser.add_argument('--visualise-tsne', default=False, action='store_true', help="If not set then use UMAP.")

args = parser.parse_args()

main(vars(args))
