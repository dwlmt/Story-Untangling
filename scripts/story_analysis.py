''' Create Analysis Charts for Stories in bulk based on the preidction output and cluster analysis.

'''
import argparse
import os

import colorlover as cl
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.offline as py

py.init_notebook_mode(connected=True)

plotly.io.orca.config.save

# These are the default plotly colours.
colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                     'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                     'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                     'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                     'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

shapes = ["circle", "square", "diamond","cross", "x","triangle-up", "triangle-down", "triangle-left", "triangle-right","pentagon","hexagon","octagon",'hexagram',"bowtie","hourglass"]

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--batch-stats', required=True, type=str, help="CSV of the prediction batch stats.")
parser.add_argument('--position-stats', required=True, type=str, help="CSV of the prediction position stats.")
parser.add_argument('--window-stats', required=True, type=str, help="CSV of the window stats.")
parser.add_argument('--vector-stats', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument('--smoothing', required=False, type=str, nargs='*', default=['exp','holt','avg','avg_2','reg','reg_2','arima'], help="CSV containing the vector output.")
parser.add_argument('--max-plot-points', default=50000, type=int, help="Max number of scatter points.")
parser.add_argument('--cluster-example-num', default=100, type=int, help="Max number of examples to select for each cluster category.")
parser.add_argument("--smoothing-plots", default=False, action="store_true" , help="Plot sliding windows and smoothong as well as the raw position data.")
parser.add_argument("--exponential-plots", default=False, action="store_true" , help="Plot exponential fit.")
parser.add_argument("--no-html-plots", default=False, action="store_true" , help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true" , help="Don't save plots to PDF")

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

    create_sentiment_plots(args)
    create_story_plots(args)

    create_cluster_examples(args)
    create_cluster_scatters(args)


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

        group = field_df.groupby(field).apply(lambda x: x.sample(min(len(x), args["cluster_example_num"]))).reset_index(drop=True)

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

                    colors = cl.scales['5']['div']['Spectral']

                    unique_column_values = field_df[cat_field].unique()
                    num_colors =  len(unique_column_values)
                    num_colors = max(num_colors, max([int(v) for v in unique_column_values]))
                    color_scale = cl.interp(colors,min(num_colors + 1,256))

                    for name, group in field_df.groupby([cat_field]):

                        if int(name) == -1:
                            series_color = 'lightslategrey'
                        else:
                            series_color = color_scale[int(name) % 255]

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
                        plotly.offline.plot(fig, filename=f"{args['output_dir']}/cluster_scatters/{field}_{cat_field}_scatter.html", auto_open=False)

                    if not args["no_pdf_plots"]:
                        pio.write_image(fig, f"{args['output_dir']}/cluster_scatters/{field}_{cat_field}_scatter.pdf")

def create_sentiment_plots(args):

    ensure_dir(f"{args['output_dir']}/sentiment_plots/")

    position_df = pd.read_csv(args["position_stats"])

    for name, group in position_df.groupby("story_id"):

        group_df = group.sort_values(by=['sentence_num'])

        data = []

        color_idx = 0
        for i, pred in enumerate(['textblob_sentiment','vader_sentiment','sentiment']):

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
                name=f'{pred}'.replace('sentiment','sent'),
            )
            data.append(trace)

            color_idx += 1


        layout = go.Layout(
            title=f'Story {name} Sentiment Plot',
            hovermode='closest',
            xaxis=dict(
                #title='Position',
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
            plotly.offline.plot(fig, filename=f"{args['output_dir']}/sentiment_plots/story_{name}_sentiment_plot.html",
                                auto_open=False)
        if not args["no_pdf_plots"]:
            pio.write_image(fig, f"{args['output_dir']}/sentiment_plots/story_{name}_sentiment_plot.pdf")


def create_story_plots(args):

    ensure_dir(f"{args['output_dir']}/prediction_plots/")

    position_df = pd.read_csv(args["position_stats"])

    prediction_columns = ['generated_surprise_l1', 'generated_surprise_l2'
        , 'generated_suspense_l1', 'generated_suspense_l2',
                          'generated_suspense_entropy'
                          'corpus_surprise_l1', 'corpus_surprise_l2',
                          'corpus_suspense_l1', 'corpus_suspense_l2',
                          'corpus_suspense_entropy',
                          'generated_surprise_l1_state', 'generated_surprise_l2_state',
                          'generated_suspense_l1_state', 'generated_suspense_l2_state',
                          'corpus_surprise_l1_state', 'corpus_surprise_l2_state',
                          'corpus_suspense_l1_state', 'corpus_suspense_l2_state']

    window_df = pd.read_csv(args["window_stats"])
    window_sizes = window_df["window_size"].unique()
    measure_names = window_df["window_name"].unique()

    vector_df = pd.read_csv(args["vector_stats"])

    for name, group in position_df.groupby("story_id"):

        group_df = group.sort_values(by=['sentence_num'])

        story_win_df = window_df.loc[window_df['story_id'] == name]

        for y_axis_group in ['l1','l2','entropy']:

            data = []

            color_idx = 0
            for i, pred in enumerate(prediction_columns):

                pred_name = pred.replace('suspense','susp').replace('surprise','surp').replace('corpus','cor').replace('generated','gen').replace('state','st')

                if y_axis_group not in pred:
                    continue

                if pred not in group_df.columns:
                    continue

                # Don't plot both corpus and generation surprise as they are the same.
                if "surprise" in pred:
                   if not "generated" in pred:
                       continue

                text = [f"<b>{t}</b>" for t in group_df["sentence_text"]]

                vector_row_df = vector_df.merge(group_df, left_on='sentence_id', right_on='sentence_id')
                for field in sentence_cluster_fields + story_cluster_fields + ['sentiment', 'vader_sentiment', 'textblob_sentiment']:
                    if field in vector_row_df.columns:
                        text = [t + f"<br>{field}: {f}" for (t, f) in zip(text, vector_row_df[field])]

                trace = go.Scatter(
                    x=group_df['sentence_num'],
                    y=group_df[pred],
                    text=text,
                    mode='lines+markers',
                    line = dict(
                        color=colors[color_idx]
                    ),
                    name=f'{pred_name}',
                )
                data.append(trace)

                if args['smoothing_plots']:
                    if pred in measure_names:
                        for j, window in enumerate(window_sizes):

                            if window not in args['smoothing']:
                                continue

                            win_df = story_win_df[['window_size','window_name','position','mean']]

                            win_df = win_df.loc[win_df['window_size'] == window]

                            win_df = win_df.loc[win_df['window_name'] == pred]
                            win_df = win_df.sort_values(by="position")


                            trace = go.Scatter(
                                x=win_df['position'],
                                y=win_df['mean'],
                                mode='lines+markers',
                                name=f'{pred_name} - {window.replace("exponential","exp")}',
                                line=dict(
                                    dash='dot',color=colors[color_idx]),
                                marker = dict(
                                    symbol=shapes[j])
                            )
                            data.append(trace)

                color_idx += 1


            layout = go.Layout(
                title=f'Story {name} Prediction Plot',
                hovermode='closest',
                xaxis=dict(
                    #title='Position',
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
                plotly.offline.plot(fig, filename=f"{args['output_dir']}/prediction_plots/story_{name}_{y_axis_group}_plot.html",
                                    auto_open=False)
            if not args["no_pdf_plots"]:
                pio.write_image(fig, f"{args['output_dir']}/prediction_plots/story_{name}_{y_axis_group}_plot.pdf")


    print(position_df.columns)


def create_analysis_output(args):

    print(args)
    
    analyse_vector_stats(args)

    ensure_dir(args["output_dir"])


create_analysis_output(vars(args))