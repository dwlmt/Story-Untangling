''' Story analysis at the lower level of analysis using the raw vectors rather than the summary.
'''
import argparse
import os

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize

plt.rc('font', size=8)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


parser = argparse.ArgumentParser(
    description='story level clustering and annotations.')
parser.add_argument('--vectors', required=True, type=str, help="Parquet store with the vectors")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument("--projection-fields", required=False, type=str, nargs="+", default=['sentence_tensor_euclidean_umap_48',
                                                                         'sentence_tensor_diff_euclidean_umap_48',
                                                                         'story_tensor_diff_euclidean_umap_48',
                                                                         'story_tensor_euclidean_umap_48',
                                                                         'sentence_tensor_cosine_umap_48',
                                                                         'sentence_tensor_diff_cosine_umap_48',
                                                                         'story_tensor_diff_cosine_umap_48',
                                                                         'story_tensor_cosine_umap_48',
                                                                         'sentence_tensor_pca_48',
                                                                         'story_tensor_pca_48',
                                                                         'sentence_tensor_diff_pca_48',
                                                                         'story_tensor_diff_pca_48'
                                                                         ])
parser.add_argument('--similarity-metric', default=["cosine", "euclidean"], nargs="+", type=str,
                    help="The similarity measure to use.")

metadata_fields = ['sentence_id', 'sentence_length', 'sentence_num', 'sentence_tensor',
                   'sentence_text', 'transition_text', 'story_id']

def create_analysis_output(args):
    ensure_dir(f"{args['output_dir']}/mst_plots/")

    print(args)

    vectors_df = dd.read_parquet(args["vectors"])

    story_ids = vectors_df["story_id"].unique().compute()

    for story_id in story_ids:

        story_df = vectors_df.loc[vectors_df['story_id'] == story_id].compute()
        story_df = story_df.sort_values(by=['sentence_num'])

        for proj_field in args['projection_fields']:

            try:
                print(f"Local clustering for: {story_id}")

                proj_df = story_df[metadata_fields + [proj_field]]

                for sim_metric in args["similarity_metric"]:

                    proj_list = proj_df[proj_field].tolist()
                    proj_arr = numpy.stack(proj_list, axis=0)

                    # Cosine is failing because it doesn't work with the large KD Trees so use angular distance instead.
                    metric = sim_metric

                    if sim_metric == "cosine":
                        metric = "euclidean"
                        proj_arr = normalize(proj_arr, norm='l2')

                    clusterer = HDBSCAN(algorithm='best', metric=metric, gen_min_span_tree=True, approx_min_span_tree=False,
                                        core_dist_n_jobs=16, min_cluster_size=3)


                    clusterer.fit(proj_arr)

                    import seaborn as sns
                    palette = sns.color_palette()
                    cluster_colors = [sns.desaturate(palette[col], sat)
                                      if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                                      zip(clusterer.labels_, clusterer.probabilities_)]

                    labels = proj_df['sentence_num']

                    reference_df = proj_df[['sentence_num', 'sentence_text']]
                    reference_df["cluster"] = clusterer.labels_
                    reference_df["probability"] = clusterer.probabilities_

                    csv_path = f'{args["output_dir"]}/mst_plots/{story_id}_{proj_field}_{sim_metric}_ref.csv'
                    print(f"Save csv: {csv_path}")
                    reference_df.to_csv(csv_path)

                    ax = clusterer.minimum_spanning_tree_.plot()

                    d = ax.collections[0]
                    d.set_offset_position('data')
                    points = d.get_offsets()
                    ymin, ymax = ax.get_ylim()

                    bonus = (ymax - ymin) / 50
                    for point, name, color in zip(points, labels, cluster_colors):
                        x, y = point
                        ax.text(x, y + bonus, name, color=color, size='smaller')
                    fig = ax.get_figure()

                    file_path = f'{args["output_dir"]}/mst_plots/{story_id}_{proj_field}_{sim_metric}.pdf'

                    print(f"Save figure: {file_path}")

                    fig.savefig(file_path)
                    fig.clear()
                    plt.close(fig)

            except Exception as e:
                print(e)


args = parser.parse_args()

create_analysis_output(vars(args))
