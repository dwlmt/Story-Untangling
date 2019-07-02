''' Create Analysis Charts for Stories in bulk based on the preidction output and cluster analysis.

'''
import argparse
import ast
import os
import re

import pandas as pd

import cufflinks as cf
import plotly.graph_objs as go
import plotly.offline as py
import plotly.figure_factory as ff
import plotly
import numpy as np

import plotly.io as pio
import numpy as np
import pandas as pd

py.init_notebook_mode(connected=True)

plotly.io.orca.config.save

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--batch-stats', required=True, type=str, help="CSV of the prediction batch stats.")
parser.add_argument('--position-stats', required=True, type=str, help="CSV of the prediction position stats.")
parser.add_argument('--window-stats', required=True, type=str, help="CSV of the window stats.")
parser.add_argument('--vector-stats', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument('--max-plot-points', default=20000, type=int, help="Max number of scatter points.")

args = parser.parse_args()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)

projection_fields = ['sentence_tensor_euclidean_umap_2', 'sentence_tensor_pca_2',
                        'story_tensor_euclidean_umap_2', 'story_tensor_pca_2']

sentence_cluster_fields = ['sentence_tensor_euclidean_umap_48_cluster_kmeans_cluster',
                               'sentence_tensor_euclidean_umap_48_cluster_product_code',
                               'sentence_tensor_pca_48_cluster_kmeans_cluster',
                               'sentence_tensor_pca_48_cluster_product_code',
                               'sentence_tensor_euclidean_umap_48_cluster_label',
                               'sentence_tensor_pca_48_cluster_label']

story_cluster_fields = ['story_tensor_euclidean_umap_48_cluster_kmeans_cluster',
                        'story_tensor_euclidean_umap_48_cluster_product_code',
                        'story_tensor_pca_48_cluster_kmeans_cluster',
                        'story_tensor_pca_48_cluster_product_code',
                        'story_tensor_euclidean_umap_48_cluster_label',
                        'story_tensor_pca_48_cluster_label']


def analyse_vector_stats(args):

    #create_cluster_examples(args)

    create_cluster_scatters(args)

    #create_story_clusters(args)


def create_cluster_examples(args):

    vector_df = pd.read_csv(args["vector_stats"])
    ensure_dir(f"{args['output_dir']}/cluster_examples/")
    for field in sentence_cluster_fields + story_cluster_fields:
        print(f"Create cluster examples for: {field}")


        fields_to_extract = [field]
        if "kmeans" in field:
            print(field)
            distance_field = field.replace("kmeans_cluster","kmeans_distance")
            fields_to_extract.append(distance_field)

        if "label" in field:
            prob_field = field.replace("_label", "_probability")
            fields_to_extract.append(prob_field)

            outlier_field = field.replace("_label", "_outlier_score")
            fields_to_extract.append(outlier_field)

        fields_to_extract += ["sentence_text", "story_id"]

        field_df = vector_df[fields_to_extract]

        product_columns = None
        if "product_code" in field:
            product_columns = [f'{field}_1', f'{field}_2', f'{field}_3', f'{field}_4']

            field_list = field_df[field].apply(lambda x: x.replace('[','').replace(' ]','').replace(']','').split()).tolist()

            split_product_codes = pd.DataFrame(field_list,
                                               columns=product_columns)
            field_df[product_columns] = split_product_codes

        group = field_df.groupby(field).apply(lambda x: x.sample(min(len(x), 50))).reset_index(drop=True)

        group.to_csv(f"{args['output_dir']}/cluster_examples/{field}.csv")


def create_cluster_scatters(args):
    vector_df = pd.read_csv(args["vector_stats"])
    ensure_dir(f"{args['output_dir']}/cluster_scatters/")

    vector_df = vector_df.sample(n=args['max_plot_points'])

    for field in projection_fields:
        print(f"Create cluster examples for: {field}")

        coord_columns=['x','y']

        fields_to_extract = [field]

        fields_to_extract += sentence_cluster_fields
        fields_to_extract += story_cluster_fields

        fields_to_extract += ["sentence_text", "story_id", 'sentence_num']

        field_df = vector_df[fields_to_extract]

        field_list = field_df[field].apply(
            lambda x: x.replace('[', '').replace(' ]', '').replace(']', '').split()).tolist()

        split_xy = pd.DataFrame(field_list,
                                           columns=coord_columns)
        field_df[coord_columns] = split_xy

        field_df['label'] = field_df.apply (lambda row: f"{row['story_id']} : {row['sentence_num']} - {row['sentence_text']}", axis=1)

        for cat_field in story_cluster_fields + sentence_cluster_fields:

            # Don't cluster for each sub product code
            if "product_code" in cat_field:
                continue

            if ("pca" in field and "pca" in cat_field) or ("umap" in field and "umap" in cat_field):

                if ("story" in field and "story" in cat_field) or ("sentence" in field and "sentence" in cat_field):

                    data = []

                    for name, group in field_df.groupby([cat_field]):
                        trace = go.Scattergl(
                            x=group['x'],
                            y=group['y'],
                            text=group['label'],
                            mode='markers',
                            name=name,
                            marker=dict(
                                #color='#FFBAD2',
                                line=dict(width=1)
                            )
                        )
                        data.append(trace)


                    plotly.offline.plot(data, filename=f"{args['output_dir']}/cluster_scatters/{field}_{cat_field}_scatter.html", auto_open=False)

def create_story_clusters(args):
    vector_df = pd.read_csv(args["vector_stats"])
    ensure_dir(f"{args['output_dir']}/story_clusters/")



def create_analysis_output(args):

    print(args)
    
    analyse_vector_stats(args)

    ensure_dir(args["output_dir"])


create_analysis_output(vars(args))