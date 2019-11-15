''' Create Analysis Charts for Stories in bulk based on the preidction output and cluster analysis.

'''
import argparse
import os
import statistics
from copy import deepcopy
from itertools import zip_longest
from math import floor, ceil
from textwrap import TextWrapper

import colorlover as cl
import numpy
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
# These are the default plotly colours.
from scipy.signal import find_peaks
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler

colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

shapes = ["circle", "square", "diamond", "cross", "x", "star-triangle-up", "star-triangle-down", "triangle-up",
          "triangle-down", "triangle-left", "triangle-right", "pentagon", "hexagon", "octagon", 'hexagram', "bowtie",
          "hourglass"]

parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--batch-stats', required=False, type=str, help="CSV of the prediction batch stats.")
parser.add_argument('--position-stats', required=False, type=str, help="CSV of the prediction position stats.")
parser.add_argument('--window-stats', required=False, type=str, help="CSV of the window stats.")
parser.add_argument('--vector-stats', required=False, type=str, help="CSV containing the vector output.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument('--smoothing', required=False, type=str, nargs='*',
                    default=['exp', 'holt', 'avg', 'avg_2', 'reg', 'reg_2', 'arima'],
                    help="CSV containing the vector output.")
parser.add_argument('--max-plot-points', default=50000, type=int, help="Max number of scatter points.")
parser.add_argument('--cluster-example-num', default=100, type=int,
                    help="Max number of examples to select for each cluster category.")
parser.add_argument("--smoothing-plots", default=False, action="store_true",
                    help="Plot sliding windows and smoothong as well as the raw position data.")
parser.add_argument("--no-html-plots", default=False, action="store_true", help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true", help="Don't save plots to PDF")
parser.add_argument("--no-cluster-output", default=False, action="store_true",
                    help="Don't calculate the cluster output.")
parser.add_argument("--no-story-output", default=False, action="store_true", help="Don't calculate the story plots.")
parser.add_argument('--peak-prominence-weighting', default=0.35, type=float,
                    help="Use to scale the standard deviation of a column.")
parser.add_argument('--peak-width', default=1.0, type=float,
                    help="How wide must a peak be to be included. 1.0 allow a single point sentence to be a peak.")
parser.add_argument('--number-of-peaks', default=-1, type=int,
                    help="Number of peaks to find, overrides the other settings.")
parser.add_argument('--turning-points-csv', required=False, type=str,
                    help="If provided the turning points to compare against from the CSV.")
parser.add_argument('--turning-point-columns', required=False, type=str, nargs="*",
                    default=["tp1", "tp2", "tp3", "tp4", "tp5"],
                    help="If provided the turning points to compare against from the CSV.")
parser.add_argument('--turning-point-means', required=False, type=float, nargs="*",
                    default=[11.39, 31.86, 50.65, 74.15, 89.43],
                    help="If turning points provided then these are the expected positions.")
parser.add_argument('--turning-point-stds', required=False, type=float, nargs="*",
                    default=[6.72, 11.26, 12.15, 8.40, 4.74],
                    help="If turning points provided then these are the expected positions.")

args = parser.parse_args()


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


projection_fields = ['sentence_tensor_euclidean_umap_2', 'sentence_tensor_pca_2', 'sentence_tensor_cosine_umap_2',
                     'story_tensor_euclidean_umap_2', 'story_tensor_pca_2', 'story_tensor_cosine_umap_2']

sentence_cluster_fields = ['sentence_tensor_euclidean_umap_48_cluster_kmeans_cluster',
                           'sentence_tensor_euclidean_umap_48_cluster_product_code',
                           'sentence_tensor_euclidean_umap_48_cluster_label',
                           'sentence_tensor_cosine_umap_48_cluster_kmeans_cluster',
                           'sentence_tensor_cosine_umap_48_cluster_product_code',
                           'sentence_tensor_cosine_umap_48_cluster_label',
                           'sentence_tensor_pca_48_cluster_kmeans_cluster',
                           'sentence_tensor_pca_48_cluster_product_code',
                           'sentence_tensor_pca_48_cluster_label']

story_cluster_fields = ['story_tensor_euclidean_umap_48_cluster_kmeans_cluster',
                        'story_tensor_euclidean_umap_48_cluster_product_code',
                        'story_tensor_euclidean_umap_48_cluster_label',
                        'story_tensor_cosine_umap_48_cluster_kmeans_cluster',
                        'story_tensor_cosine_umap_48_cluster_product_code',
                        'story_tensor_cosine_umap_48_cluster_label',
                        'story_tensor_pca_48_cluster_label',
                        'story_tensor_pca_48_cluster_kmeans_cluster',
                        'story_tensor_pca_48_cluster_product_code',
                        ]

# Add the diff (delta) sentence to sentence fields to fields to be processed.

for fields in [projection_fields, sentence_cluster_fields, story_cluster_fields]:
    join_fields = []
    for field in fields:
        if "story_tensor" in field:
            join_fields.append(field.replace("story_tensor", "story_tensor_diff"))
        if "sentence_tensor" in field:
            join_fields.append(field.replace("sentence_tensor", "sentence_tensor_diff"))

    fields.extend(join_fields)


def analyse_vector_stats(args):
    if not args["no_story_output"]:
        create_sentiment_plots(args)
        create_story_plots(args)

    if not args["no_cluster_output"]:
        create_cluster_examples(args)
        create_cluster_scatters(args)


def create_cluster_examples(args):

    if args["vector_stats"] is None or len(args["vector_stats"]) == 0:
        return

    metadata_fields = ["story_id", 'sentence_num', 'sentence_id', "sentence_text", "transition_text"]

    vector_df = pd.read_csv(args["vector_stats"])

    ensure_dir(f"{args['output_dir']}/cluster_examples/")
    for field in sentence_cluster_fields + story_cluster_fields:

        fields_to_save = []

        print(f"Create cluster examples for: {field}")

        fields_to_extract = []

        if "kmeans" in field:
            print(field)
            distance_field = field.replace("kmeans_cluster", "kmeans_distance")
            fields_to_extract.append(distance_field)

        if "label" in field:
            prob_field = field.replace("_label", "_probability")
            fields_to_extract.append(prob_field)

            outlier_field = field.replace("_label", "_outlier_score")
            fields_to_extract.append(outlier_field)

        fields_to_extract += metadata_fields

        product_columns = None
        if "product_code" in field:
            product_columns = [f'{field}_1', f'{field}_2', f'{field}_3', f'{field}_4']

            field_list = vector_df[field].apply(
                lambda x: [int(x) for x in x.replace('[', '').replace(' ]', '').replace(']', '').split()]).tolist()
            field_list = [[sent_id] + l for l, sent_id in zip(field_list, vector_df['sentence_id'])]

            split_product_codes = pd.DataFrame(field_list,
                                               columns=["sentence_id"] + product_columns)

            vector_df = vector_df.merge(split_product_codes, left_on='sentence_id', right_on='sentence_id')

            fields_to_save += product_columns
        else:
            fields_to_save.append(field)

        for field_to_save in fields_to_save:
            fields_to_extract.append(field_to_save)

            field_df = vector_df[fields_to_extract]

            group = field_df.groupby(field_to_save).apply(
                lambda x: x.sample(min(len(x), args["cluster_example_num"]))).reset_index(drop=True)
            file_path = f"{args['output_dir']}/cluster_examples/{field_to_save}.csv"
            print(f"Save examples: {file_path}")
            group.to_csv(file_path)


def create_cluster_scatters(args):

    if  args["vector_stats"] is None  or len(args["vector_stats"]) == 0:
        return

    vector_df = pd.read_csv(args["vector_stats"])
    ensure_dir(f"{args['output_dir']}/cluster_scatters/")

    vector_df = vector_df.sample(n=min(args['max_plot_points'], len(vector_df)))

    product_fields = []
    for proj_field in story_cluster_fields + sentence_cluster_fields:
        if "product_code" in proj_field:
            product_columns = [f'{proj_field}_1', f'{proj_field}_2', f'{proj_field}_3', f'{proj_field}_4']

            field_list = vector_df[proj_field].apply(
                lambda x: x.replace('[', '').replace(' ]', '').replace(']', '').split()).tolist()

            field_list = [[sent_id] + l for l, sent_id in zip(field_list, vector_df['sentence_id'])]

            split_product_codes = pd.DataFrame(field_list,
                                               columns=["sentence_id"] + product_columns)

            vector_df = vector_df.merge(split_product_codes, left_on='sentence_id', right_on='sentence_id')

            product_fields += product_columns

    for field in projection_fields:
        print(f"Create cluster examples for: {field}")

        coord_columns = ['x', 'y']

        fields_to_extract = [field]

        fields_to_extract += sentence_cluster_fields
        fields_to_extract += story_cluster_fields
        fields_to_extract += product_fields
        fields_to_extract += ["sentence_text"]
        fields_to_extract += ["transition_text"]

        fields_to_extract += ["story_id", 'sentence_num']

        field_df = vector_df[fields_to_extract]

        field_list = field_df[field].apply(
            lambda x: x.replace('[', '').replace(' ]', '').replace(']', '').split()).tolist()

        split_xy = pd.DataFrame(field_list,
                                columns=coord_columns)
        field_df[coord_columns] = split_xy

        for cluster_field in product_fields + story_cluster_fields + sentence_cluster_fields:

            if cluster_field.endswith("product_code"):
                continue

            if "pca" in field and not "pca" in cluster_field:
                continue
            if "cosine" in field and not "cosine" in cluster_field:
                continue
            if "euclidean" in field and not "euclidean" in cluster_field:
                continue

            if ("pca" in field and "pca" in cluster_field) or ("umap" in field and "umap" in cluster_field):

                if ("story" in field and "story" in cluster_field) or (
                        "sentence" in field and "sentence" in cluster_field):

                    if not "diff" in cluster_field:
                        text_field = 'sentence_text'
                    else:
                        text_field = 'transition_text'

                    field_df['label'] = field_df.apply(
                        lambda
                            row: f"<b>{row[text_field]}</b> <br>cluster: {row[cluster_field]} <br>story_id: {row['story_id']} <br>sentence_num: {row['sentence_num']}",
                        axis=1)

                    data = []

                    colors = cl.scales['5']['div']['Spectral']

                    unique_column_values = field_df[cluster_field].unique()
                    num_colors = len([u for u in unique_column_values if u != -1])
                    num_colors = max(num_colors, max([int(v) for v in unique_column_values]))
                    num_colors = min(num_colors, 64)

                    if num_colors > 5:
                        color_scale = cl.interp(colors, num_colors)
                    else:
                        color_scale = colors

                    for name, group in field_df.groupby([cluster_field]):

                        if int(name) == -1:
                            series_color = 'lightslategrey'
                        else:
                            series_color = color_scale[int(name) % num_colors]

                        trace = go.Scattergl(
                            x=group['x'],
                            y=group['y'],
                            text=group['label'],
                            mode='markers',
                            name=name,
                            marker=dict(
                                line=dict(width=1),
                                color=[series_color] * len(group['x']),
                                symbol=shapes[int(name) % len(shapes)]
                            )
                        )
                        data.append(trace)

                    layout = dict()

                    fig = dict(data=data, layout=layout)

                    if not args["no_html_plots"]:
                        file_path = f"{args['output_dir']}/cluster_scatters/{field}_{cluster_field}_scatter.html"
                        print(f"Save plot: {file_path}")
                        pio.write_html(fig, file_path)

                    if not args["no_pdf_plots"]:
                        file_path = f"{args['output_dir']}/cluster_scatters/{field}_{cluster_field}_scatter.pdf"
                        print(f"Save plot pdf: {file_path}")
                        pio.write_image(fig, file_path)


def create_sentiment_plots(args):

    if  args["position_stats"] is None or len(args["position_stats"]) == 0:
        return

    ensure_dir(f"{args['output_dir']}/sentiment_plots/")

    position_df = pd.read_csv(args["position_stats"])


    for name, group in position_df.groupby("story_id"):

        group_df = group.sort_values(by=['sentence_num'])

        data = []

        color_idx = 0
        for i, pred in enumerate(['textblob_sentiment', 'vader_sentiment', 'sentiment']):

            # Don't plot both corpus and generation surprise as they are the same.
            if "surprise" in pred:
                if not "generated" in pred:
                    continue

            text = [f"<b>{t}</b>" for t in group_df["sentence_text"]]

            trace = go.Scatter(
                x=group_df['sentence_num'],
                y=group_df[pred],
                text=text,
                mode='lines+markers',
                line=dict(
                    color=colors[color_idx]
                ),
                name=f'{pred}'.replace('sentiment', 'sent'),
            )
            data.append(trace)

            color_idx += 1

        layout = go.Layout(
            title=f'Story {name} Sentiment Plot',
            hovermode='closest',
            xaxis=dict(
                # title='Position',
            ),
            yaxis=dict(
                title='Sentiment',
            ),
            showlegend=True,
            legend=dict(
                orientation="h")
        )

        fig = go.Figure(data=data, layout=layout)

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/sentiment_plots/story_{name}_sentiment_plot.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)
        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/sentiment_plots/story_{name}_sentiment_plot.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)

prediction_columns = ["generated_surprise_word_overlap",
                          "generated_surprise_simple_embedding",
                          'generated_surprise_l1', 'generated_surprise_l2'
                          , 'generated_suspense_l1', 'generated_suspense_l2',
                          'generated_suspense_entropy',
                          'corpus_suspense_entropy',
                          'generated_surprise_entropy',
                          "corpus_surprise_word_overlap",
                          "corpus_surprise_simple_embedding",
                          'corpus_surprise_entropy',
                          'corpus_surprise_l1', 'corpus_surprise_l2',
                          'corpus_suspense_l1', 'corpus_suspense_l2',
                          'generated_surprise_l1_state', 'generated_surprise_l2_state',
                          'generated_suspense_l1_state', 'generated_suspense_l2_state',
                          'corpus_surprise_l1_state', 'corpus_surprise_l2_state',
                          'corpus_suspense_l1_state', 'corpus_suspense_l2_state',
                          'textblob_sentiment', 'vader_sentiment', 'sentiment']

def scale_prediction_columns(position_df):
    for col in prediction_columns:

        scaler = StandardScaler()
        scaled_col = numpy.squeeze(scaler.fit_transform(position_df[col].to_numpy().reshape(-1, 1)), axis=1).tolist()
        position_df[f"{col}_scaled"] = scaled_col
        print(position_df[f"{col}_scaled"], scaled_col)

    position_df = position_df.reset_index()

    return position_df

def create_story_plots(args):
    if  args["position_stats"] is None or len(args["position_stats"]) == 0:
        return

    ensure_dir(f"{args['output_dir']}/prediction_plots/")

    turning_points_df = None
    if "turning_points_csv" in args and args["turning_points_csv"] is not None:
        turning_points_df = pd.read_csv(args["turning_points_csv"])
        turning_points_df = turning_points_df.fillna(value=0.0)

    position_df = pd.read_csv(args["position_stats"])

    position_df = scale_prediction_columns(position_df)

    print(position_df.columns)

    position_df = position_df.fillna(value=0.0)

    window_df = pd.read_csv(args["window_stats"])
    window_sizes = window_df["window_size"].unique()
    measure_names = window_df["window_name"].unique()

    if args["vector_stats"] is not None:
        vector_df = pd.read_csv(args["vector_stats"])
    else:
        vector_df = None

    y_axis_map = {}

    segmented_data = []

    for y_axis_group in ['l1', 'l2', 'entropy', 'baseline','scaled']:

        column_list = []

        for i, pred in enumerate(prediction_columns):

            if  y_axis_group != "scaled" and y_axis_group not in pred and y_axis_group != "baseline" or (
                    y_axis_group is "baseline" and "overlap" not in pred and "embedding" not in pred):
                continue

            if y_axis_group != "scaled":
                column_list.append(pred)
            else:
                column_list.append(f"{pred}_scaled")

        y_axis_map[y_axis_group] = column_list

    turning_point_data_list = []

    for story_id, group in position_df.groupby("story_id"):

        group_df = group.sort_values(by=['sentence_num'])

        print(group_df)

        story_win_df = window_df.loc[window_df['story_id'] == story_id]

        for y_axis_group, y_axis_columns in y_axis_map.items():

            print(y_axis_columns)

            plotted_turning_points = False

            data = []

            prom_data = []
            for c in y_axis_columns:
                prom_data.extend(group_df[c].tolist())

            prom_weighting = args["peak_prominence_weighting"]
            if y_axis_group == "scaled":
                print(prom_data,prom_weighting)
            if y_axis_group == "scaled":
                prominence_threshold =  prom_weighting
            else:
                prominence_threshold = statistics.stdev(prom_data) * prom_weighting

            if args["number_of_peaks"] > 0:
                prominence_threshold = 0.00001  # Make tiny as is not specified then the metadata is not returned for prominences.

            color_idx = 0

            max_point = 0.0
            for i, pred in enumerate(y_axis_columns):

                pred_data = group_df[pred].tolist()
                if y_axis_group == "scaled":

                    measure_offset = 0.0 - pred_data[0]
                    pred_data = [m + measure_offset for m in pred_data]

                pred_name = pred.replace('suspense', 'susp').replace('surprise', 'surp').replace('corpus',
                                                                                                 'cor').replace(
                    'generated', 'gen').replace('state', 'st').replace('_scaled','')

                # Don't plot both corpus and generation surprise as they are the same.
                if "surprise" in pred:
                    if not "generated" in pred and "entropy" not in pred and y_axis_group != "baseline":
                        continue

                text = [f"<b>{t}</b>" for t in group_df["sentence_text"]]

                if vector_df is not None:
                    vector_row_df = vector_df.merge(group_df, left_on='sentence_id', right_on='sentence_id')
                    for field in sentence_cluster_fields + story_cluster_fields + ['sentiment', 'vader_sentiment',
                                                                                   'textblob_sentiment']:
                        if field in vector_row_df.columns and len(vector_row_df[field]) > 0:
                            text = [t + f"<br>{field}: {f}" for (t, f) in zip(text, vector_row_df[field])]

                max_point = max(max_point, max(group_df[pred]))

                trace = go.Scatter(
                    x=group_df['sentence_num'],
                    y=pred_data,
                    text=text,
                    mode='lines+markers',
                    line=dict(
                        color=colors[color_idx % len(colors)]
                    ),
                    name=f'{pred_name}',
                )
                data.append(trace)

                sentence_nums = group_df["sentence_num"].tolist()
                sentence_text = group_df["sentence_text"].tolist()
                y = pred_data

                type = "peak"
                peak_indices, peaks_meta = find_peaks(y, prominence=prominence_threshold, width=args["peak_width"],
                                                      plateau_size=1)

                num_of_peaks = args["number_of_peaks"]
                peak_indices = optionally_top_n_peaks(num_of_peaks, peak_indices, peaks_meta)

                if len(peak_indices) > 0:
                    hover_text, peaks_data = create_peak_text_and_metadata(peak_indices, peaks_meta, sentence_nums,
                                                                           sentence_text, story_id, "peak",
                                                                           y_axis_group, pred)

                    segmented_data.extend(peaks_data)

                    trace = go.Scatter(
                        x=[sentence_nums[j] for j in peak_indices],
                        y=[y[j] for j in peak_indices],
                        mode='markers',
                        marker=dict(
                            color=colors[color_idx % len(colors)],
                            symbol='star-triangle-up',
                            size=14,
                        ),
                        name=f'{pred_name} - {type}',
                        text=hover_text
                    )
                    data.append(trace)

                type = "trough"
                y_inverted = [x_n * -1.0 for x_n in y]
                # prominence=prom, width=args["peak_width"]
                if len(peak_indices) > 0:
                    peak_indices, peaks_meta = find_peaks(y_inverted, prominence=prominence_threshold,
                                                          width=args["peak_width"], plateau_size=1)

                    peak_indices = optionally_top_n_peaks(num_of_peaks, peak_indices, peaks_meta)

                    hover_text, peaks_data = create_peak_text_and_metadata(peak_indices, peaks_meta, sentence_nums,
                                                                           sentence_text, story_id, "trough",
                                                                           y_axis_group, pred)

                    segmented_data.extend(peaks_data)

                    trace = go.Scatter(
                        x=[sentence_nums[j] for j in peak_indices],
                        y=[y[j] for j in peak_indices],
                        mode='markers',
                        marker=dict(
                            color=colors[color_idx % len(colors)],
                            symbol='star-triangle-down',
                            size=14,
                        ),
                        name=f'{pred_name} - {type}',
                        text=hover_text
                    )
                    data.append(trace)

                all_points = []
                if turning_points_df is not None:

                    turning_story_df = turning_points_df.loc[turning_points_df["story_id"] == story_id]
                    if len(turning_story_df) > 0:

                        all_points = []

                        story_length = len(sentence_nums)
                        expected_points_indices = []

                        mean_expected = []
                        lower_brackets = []
                        upper_brackets = []
                        for mean, std in zip(args["turning_point_means"], args["turning_point_stds"]):
                            mean_pos = int(round(mean * (story_length / 100)))
                            mean_expected.append(mean_pos)
                            lower_bracket = max(0, int(floor(mean - std) * (story_length / 100)))
                            lower_brackets.append(lower_bracket)
                            upper_bracket = min(story_length - 1, int(ceil(mean + std) * (story_length / 100)))
                            upper_brackets.append(upper_bracket)

                            sentence_whole = group_df[pred].tolist()

                            index_pos = sentence_whole.index(max(sentence_whole[lower_bracket:upper_bracket + 1]))
                            expected_points_indices.append(index_pos)

                        done_story = False
                        for i, ann_point in turning_story_df.iterrows():

                            if done_story:
                                continue
                            else:
                                done_story

                            turn_dict = {}

                            turn_dict["story_id"] = story_id
                            turn_dict["measure"] = pred
                            turn_dict["annotator"] = i

                            turn_dict["lower_brackets"] = lower_brackets
                            turn_dict["upper_brackets"] = upper_brackets
                            turn_dict["mean_expected"] = mean_expected

                            annotated_points = []
                            for col in args["turning_point_columns"]:
                                point = ann_point[col]
                                annotated_points.append(point)

                            all_points.extend(annotated_points)

                            if len(expected_points_indices) == len(args["turning_point_columns"]):
                                exp_dict = deepcopy(turn_dict)

                                calc_turning_point_distances(annotated_points, args, expected_points_indices,
                                                             sentence_nums,
                                                             exp_dict, type="constrained", compared="annotated")
                                turning_point_data_list.append(exp_dict)

                            if len(peak_indices) == len(args["turning_point_columns"]):
                                peak_dict = deepcopy(turn_dict)
                                calc_turning_point_distances(annotated_points, args, peak_indices, sentence_nums,
                                                             peak_dict, type="unconstrained", compared="annotated")
                                turning_point_data_list.append(peak_dict)

                        if len(peak_indices) == len(args["turning_point_columns"]):
                            exp_dict = deepcopy(turn_dict)
                            calc_turning_point_distances(mean_expected, args, expected_points_indices, sentence_nums,
                                                         exp_dict, type="constrained", compared="dist_baseline")
                            turning_point_data_list.append(exp_dict)

                        if len(peak_indices) == len(args["turning_point_columns"]):
                            peak_dict = deepcopy(turn_dict)
                            calc_turning_point_distances(mean_expected, args, peak_indices, sentence_nums,
                                                         peak_dict, type="unconstrained", compared="dist_baseline")
                            turning_point_data_list.append(peak_dict)

                    if len(expected_points_indices) > 0:
                        trace = go.Scatter(
                            x=[sentence_nums[j] for j in expected_points_indices],
                            y=[y[j] for j in expected_points_indices],
                            mode='markers',
                            marker=dict(
                                color=colors[color_idx % len(colors)],
                                symbol='triangle-up',
                                size=14,
                            ),
                            name=f'{pred_name} - {type} constrained',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in expected_points_indices],
                        )
                        data.append(trace)

                    if len(annotated_points) > 0 and not plotted_turning_points:
                        plotted_turning_points = True

                        trace = go.Scatter(
                            x=[sentence_nums[p] for p in mean_expected if p < len(sentence_nums)],
                            y=[0.0] * len(mean_expected),
                            mode='markers',
                            marker=dict(
                                color="black",
                                symbol='diamond',
                                size=14,
                            ),
                            name=f'dist baseline',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in mean_expected if
                                  j < len(sentence_nums)],

                        )
                        data.append(trace)

                        trace = go.Scatter(
                            x=[sentence_nums[p] for p in lower_brackets if p < len(sentence_nums)],
                            y=[0.0] * len(lower_brackets),
                            mode='markers',
                            marker=dict(
                                color="black",
                                symbol='triangle-right',
                                size=14,
                            ),
                            name=f'dist baseline lower',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in lower_brackets if
                                  j < len(sentence_nums)],

                        )
                        data.append(trace)

                        trace = go.Scatter(
                            x=[sentence_nums[p] for p in upper_brackets if p < len(sentence_nums)],
                            y=[0.0] * len(upper_brackets),
                            mode='markers',
                            marker=dict(
                                color="black",
                                symbol='triangle-left',
                                size=14,
                            ),
                            name=f'dist baseline upper',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in upper_brackets if
                                  j < len(sentence_nums)],

                        )
                        data.append(trace)

                        trace = go.Scatter(
                            x=[sentence_nums[p] for p in all_points if p < len(sentence_nums)],
                            y=[0.0] * len(all_points),
                            mode='markers',
                            marker=dict(
                                color="gold",
                                symbol='star',
                                size=14,
                            ),
                            name=f'annotated',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in all_points if
                                  j < len(sentence_nums)],

                        )
                        data.append(trace)

                if args['smoothing_plots']:
                    if pred in measure_names:
                        for j, window in enumerate(window_sizes):

                            if window not in args['smoothing']:
                                continue

                            win_df = story_win_df[['window_size', 'window_name', 'position', 'mean']]

                            win_df = win_df.loc[win_df['window_size'] == window]

                            win_df = win_df.loc[win_df['window_name'] == pred]
                            win_df = win_df.sort_values(by="position")

                            trace = go.Scatter(
                                x=win_df['position'],
                                y=win_df['mean'],
                                mode='lines+markers',
                                name=f'{pred_name} - {window.replace("exponential", "exp")}',
                                line=dict(
                                    dash='dot', color=colors[color_idx]),
                                marker=dict(
                                    symbol=shapes[j])
                            )
                            data.append(trace)

                color_idx += 1

            layout = go.Layout(
                title=f'Story {story_id} Prediction Plot',
                hovermode='closest',
                xaxis=dict(
                    # title='Position',
                ),
                yaxis=dict(
                    title=f'{y_axis_group}',
                ),

                showlegend=True,
                legend=dict(
                    orientation="h")
            )

            fig = go.Figure(data=data, layout=layout)

            if not args["no_html_plots"]:
                file_path = f"{args['output_dir']}/prediction_plots/story_{story_id}_{y_axis_group}_plot.html"
                print(f"Save plot {file_path}")
                pio.write_html(fig, file_path)
            if not args["no_pdf_plots"]:
                file_path = f"{args['output_dir']}/prediction_plots/story_{story_id}_{y_axis_group}_plot.pdf"
                print(f"Save plot pdf: {file_path}")
                pio.write_image(fig, file_path)

    segmented_data = pd.DataFrame(data=segmented_data)
    segmented_data.to_csv(f"{args['output_dir']}/prediction_plots/peaks_and_troughs.csv")

    if len(turning_point_data_list) > 0:
        ensure_dir(f"{args['output_dir']}/turning_points/")
        turning_point_eval_df = pd.DataFrame(data=turning_point_data_list)
        turning_point_eval_df = turning_point_eval_df.fillna(value=0.0)

        summary_data_list = []

        for group_by, group in turning_point_eval_df.groupby(by=["measure", "constraint_type", "compared"]):

            measure, constraint_type, compared = group_by

            summary_dict = {}

            summary_dict["measure"] = measure
            summary_dict["constraint_type"] = constraint_type
            summary_dict["compared"] = compared
            summary_dict["num_of_stories_counted"] = len(group)

            summary_dict["total_agreement"] = group["total_agreement"].mean()
            summary_dict["partial_agreement"] = group["partial_agreement"].sum() / len(group["partial_agreement"])

            predict_all = []
            actual_all = []
            for col in args["turning_point_columns"]:

                summary_dict[f"{col}_total_agreement"] = group[f"{col}_total_agreement_correct"].sum() / len(
                    group[f"{col}_total_agreement_correct"])

                predicted_positions = group[f"{col}_predicted_relative_position"].tolist()
                predict_all.extend(predicted_positions)
                actual_positions = group[f"{col}_expected_relative_position"].tolist()
                actual_all.extend(actual_positions)

                if len(predicted_positions) >= 2 and len(actual_positions) >= 2:
                    kendall, kendall_p_value = kendalltau(predicted_positions, actual_positions)
                    spearman, spearman_p_value = spearmanr(predicted_positions, actual_positions)
                    pearson, pearson_p_value = pearsonr(predicted_positions, actual_positions)

                    summary_dict[f"{col}_predicted_relative_position_corr_kendall"] = kendall
                    summary_dict[f"{col}_predicted_relative_position_corr_kendall_p_value"] = kendall_p_value
                    summary_dict[f"{col}_predicted_relative_position_corr_spearman"] = spearman
                    summary_dict[f"{col}_predicted_relative_position_corr_spearman_p_value"] = spearman_p_value
                    summary_dict[f"{col}_predicted_relative_position_corr_pearson"] = pearson
                    summary_dict[f"{col}_predicted_relative_position_corr_pearson_p_value"] = pearson_p_value

            if len(predicted_positions) >= 2 and len(actual_positions) >= 2:
                kendall, kendall_p_value = kendalltau(predict_all, actual_all)
                spearman, spearman_p_value = spearmanr(predict_all, actual_all)
                pearson, pearson_p_value = pearsonr(predict_all, actual_all)

                summary_dict[f"predicted_relative_position_all_corr_kendall"] = kendall
                summary_dict[f"predicted_relative_position_all_corr_kendall_p_value"] = kendall_p_value
                summary_dict[f"predicted_relative_position_all_corr_spearman"] = spearman
                summary_dict[f"predicted_relative_position_all_corr_spearman_p_value"] = spearman_p_value
                summary_dict[f"predicted_relative_position_all_corr_pearson"] = pearson
                summary_dict[f"predicted_relative_position_all_corr_pearson_p_value"] = pearson_p_value

            for c in args["turning_point_columns"] + ["avg"]:

                for d in ["dist", "norm_dist", "abs_dist", "abs_norm_dist"]:
                    col_series = group[f"{c}_{d}"]

                    col_stats = col_series.describe()

                    summary_dict[f"{c}_{d}_mean"] = col_stats["mean"]
                    summary_dict[f"{c}_{d}_std"] = col_stats["std"]
                    summary_dict[f"{c}_{d}_min"] = col_stats["min"]
                    summary_dict[f"{c}_{d}_max"] = col_stats["max"]
                    summary_dict[f"{c}_{d}_25"] = col_stats["25%"]
                    summary_dict[f"{c}_{d}_50"] = col_stats["50%"]
                    summary_dict[f"{c}_{d}_75"] = col_stats["75%"]

            summary_data_list.append(summary_dict)

        turning_point_eval_df.to_csv(f"{args['output_dir']}/turning_points/turning_point_eval_all.csv")
        turning_point_summary_df = pd.DataFrame(data=summary_data_list)

        turning_point_summary_df.to_csv(f"{args['output_dir']}/turning_points/summary_evaluation.csv")

        # Calculate summary stats.


def calc_turning_point_distances(annotated_points, args, peak_indices, sentence_nums, turn_dict, type="unconstrained",
                                 compared="annotated"):
    turn_dict["constraint_type"] = type
    turn_dict["compared"] = compared

    num_of_sentences = max(sentence_nums)

    for col, p in zip(args["turning_point_columns"], peak_indices):
        turn_dict[f"{col}_predicted_relative_position"] = p / len(sentence_nums)
        turn_dict[f"{col}_predicted_position"] = p

    for col, p in zip(args["turning_point_columns"], annotated_points):
        turn_dict[f"{col}_expected_relative_position"] = p / len(sentence_nums)
        turn_dict[f"{col}_expected_position"] = p

    points_set = set(annotated_points)
    peak_indices_set = set(peak_indices)

    for col, pred, exp in zip(args["turning_point_columns"], peak_indices, annotated_points):
        turn_dict[f"{col}_total_agreement_correct"] = int(pred == exp)

    turn_dict["total_agreement"] = len(peak_indices_set.intersection(points_set)) / len(points_set)
    turn_dict["partial_agreement"] = int(len(peak_indices_set.intersection(points_set)) > 0)
    turn_dict["annotated_total"] = len(points_set)

    distances = []
    norm_distances = []
    abs_distances = []
    abs_norm_distances = []

    for predicted, actual in zip_longest(peak_indices, annotated_points, fillvalue=peak_indices[-1]):
        distance = predicted - actual
        distances.append(distance)

        norm_distances.append(distance / float(num_of_sentences))
        abs_distance = abs(distance)
        abs_distances.append(abs_distance)
        abs_norm_distances.append(abs_distance / float(num_of_sentences))

    turn_dict[f"avg_abs_norm_dist"] = sum(abs_norm_distances) / len(abs_norm_distances)
    turn_dict[f"avg_abs_dist"] = sum(abs_distances) / len(abs_distances)
    turn_dict[f"avg_dist"] = sum(distances) / len(distances)
    turn_dict[f"avg_norm_dist"] = sum(norm_distances) / len(norm_distances)

    for norm, abs_dist, norm_dist, dist, col in zip(abs_norm_distances, abs_distances, norm_distances, distances,
                                                    args["turning_point_columns"]):
        turn_dict[f"{col}_abs_norm_dist"] = norm
        turn_dict[f"{col}_abs_dist"] = abs_dist
        turn_dict[f"{col}_norm_dist"] = norm_dist
        turn_dict[f"{col}_dist"] = dist


def optionally_top_n_peaks(num_of_peaks, peak_indices, peaks_meta):
    if num_of_peaks > 0 and len(peak_indices) > 0:
        proms = peaks_meta["prominences"]

        top_peaks = numpy.argsort(-numpy.array(proms))[:num_of_peaks]

        top_peaks = sorted(top_peaks)

        peak_indices = [peak_indices[i] for i in top_peaks]

        for col in "prominences", "right_bases", "left_bases", "widths", "width_heights", "left_ips", "right_ips", "plateau_sizes", "left_edges", "right_edges":
            meta_col = peaks_meta[col]
            peaks_meta[col] = [meta_col[i] for i in top_peaks]

    return peak_indices


def create_peak_text_and_metadata(peak_indices, peaks_meta, sentence_nums, sentence_text, story_id, type, group, field):
    hover_text = []
    peaks_list = []

    for i, ind in enumerate(peak_indices):

        peak_dict = {}

        peak_dict["story_id"] = story_id
        peak_dict["field"] = field
        peak_dict["group"] = group
        peak_dict["text"] = []
        peak_dict["sentence_nums"] = []

        left_base = peaks_meta["left_bases"][i]
        right_base = peaks_meta["right_bases"][i]
        text = ""

        for j in range(left_base, right_base):
            j = min(max(j, 0), len(sentence_nums) - 1)
            wrapper = TextWrapper(initial_indent="<br>", width=80)

            if j == ind:
                peak_dict["sentence"] = sentence_text[j]
                peak_dict["sentence_num"] = sentence_nums[j]

                wrapper = TextWrapper(initial_indent="<br>")
                wrapped_text = wrapper.fill(f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>")

                text += wrapped_text
            else:

                wrapped_text = wrapper.fill(f"{sentence_nums[j]} - {sentence_text[j]}")

                text += wrapped_text

            peak_dict["text"].append(sentence_text[j])
            peak_dict["sentence_nums"].append(sentence_nums[j])

        prominance = peaks_meta["prominences"][i]
        width = peaks_meta["widths"][i]
        importance = prominance * width

        peak_dict["prominence"] = prominance
        peak_dict["width"] = width
        peak_dict["importance"] = importance
        peak_dict["type"] = type

        text += "<br>"
        text += f"<br>Prominence: {prominance} <br>Width: {width} <br>Importance: {importance}"

        peaks_list.append(peak_dict)
        hover_text.append(text)
    return hover_text, peaks_list


def create_analysis_output(args):
    print(args)

    ensure_dir(args["output_dir"])
    analyse_vector_stats(args)


create_analysis_output(vars(args))
