import argparse
import csv
import os
import random
import re

import numpy
import pandas
from scipy.signal import find_peaks, peak_prominences, find_peaks_cwt
from scipy.stats import kendalltau, pearsonr, spearmanr
from statsmodels.compat import scipy

import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

px.defaults.height = 1000

sentiment_columns = ["sentiment", "textblob_sentiment","vader_sentiment"]

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--batch-stats', required=True, type=str, help="CSV of the prediction batch stats.")
parser.add_argument('--position-stats', required=True, type=str, help="The per sentence prediction output.")
parser.add_argument('--annotation-stats', required=True, nargs='+', type=str, help="CSV of the prediction batch stats.")
parser.add_argument("--no-html-plots", default=False, action="store_true" , help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true" , help="Don't save plots to PDF")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")

args = parser.parse_args()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)

annotation_stats_columns = ['Answer.doxaResonance',
       'Answer.doxaSurprise', 'Answer.doxaSuspense',
       'Answer.readerEmotionalResonance',
       'Answer.readerSurprise', 'Answer.readerSuspense',
       'Answer.storyInterest', 'Answer.storySentiment',
]
annotation_story_id_column = 'Input.story_id'

def label_bucket(row, attribute="measure"):
    to_label = row[attribute]
    if to_label in sentiment_columns:
        return "sentiment"
    elif "_l1" in to_label:
        return "l1"
    elif "_l2" in to_label:
        return "l2"
    elif "_l1" in to_label:
        return "l1"
    elif "_entropy" in to_label:
        return "entropy"
    else:
        return "other"

def prediction_peaks(args, annotation_df, position_df):
    ensure_dir(f"{args['output_dir']}/prediction_peaks/")

    ensure_dir(f"{args['output_dir']}/prediction_peaks/self_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_peaks/annotation_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_peaks/annotation_correlation/heatmap/")
    ensure_dir(f"{args['output_dir']}/prediction_peaks/annotation_correlation/scatter/")
    ensure_dir(f"{args['output_dir']}/prediction_peaks/multi/")
    ensure_dir(f"{args['output_dir']}/prediction_peaks/scatter/")
    ensure_dir(f"{args['output_dir']}/prediction_peaks/box/")

    position_df["sentence_text"] = position_df["sentence_text"].astype(str)

    peaks_list=[]
    peaks_summary_list = []

    story_ids = position_df["story_id"].unique()
    columns = extract_position_measure_columns(position_df)
    columns = list(set(columns).difference({"sentence_num"}))
    columns.sort()

    for story_id in story_ids:

        peak_story_row = position_df.loc[position_df["story_id"] == story_id]
        peak_story_row =  peak_story_row.sort_values(by=["sentence_num"])

        for m in columns:
            #print(story_df[["sentence_num","sentence_text",c]])

            x = peak_story_row[m].to_list()

            peak_list, summary_dict = find_and_extract_peaks(peak_story_row, "peak", x, m)

            peaks_list.extend(peak_list)
            peaks_summary_list.append(summary_dict)

            x_inverted = [x_n * -1.0 for x_n in x]
            trough_list, summary_dict = find_and_extract_peaks(peak_story_row, "trough", x_inverted, m)
            peaks_list.extend(trough_list)
            peaks_summary_list.append(summary_dict)

    peaks_df = pandas.DataFrame(peaks_list)
    peaks_df.to_csv(f"{args['output_dir']}/prediction_peaks/peaks.csv")
    print(peaks_df.columns)

    peaks_summary_df = pandas.DataFrame(peaks_summary_list)
    peaks_summary_df.to_csv(f"{args['output_dir']}/prediction_peaks/peaks_summary.csv")
    print(peaks_summary_df.columns)

    peaks_df['c'] = peaks_df.apply(lambda row: label_bucket(row), axis=1)
    peaks_summary_df['c'] = peaks_summary_df.apply(lambda row: label_bucket(row), axis=1)

    for c in peaks_df['c'].unique():
        display_df = peaks_df.loc[peaks_df['c'] == c]
        for m in ["prominences", "widths"]:

            fig = px.box(display_df, y=m, x="measure", color="type", notched=True)
            export_plots(args, f"/prediction_peaks/box/{m}_{c}_box", fig)

        fig = px.scatter(display_df, x="prominences", y="widths", color="type",
                         hover_name="sentence_text")
        export_plots(args, f"/prediction_peaks/scatter/{c}_peak", fig)

    # Calculate summary statistics for the peaks

    for c in peaks_summary_df['c'].unique():
        display_df = peaks_summary_df.loc[peaks_summary_df['c'] == c]
        for m in ["num_of_peaks", "prominence", "width", "importance"]:
            fig = px.box(display_df, y=m, x="measure", color="type", notched=True)
            export_plots(args, f"/prediction_peaks/box/{m}_{c}_summary", fig)

        for m in ["peak", "trough"]:
            bubble_df = display_df.loc[display_df["type"] == m]
            fig = px.scatter(bubble_df, x="num_of_peaks", y="width", size="prominence", color="measure", size_max=100, hover_name="story_id")
            export_plots(args, f"/prediction_peaks/scatter/{c}_{m}_summary", fig)

    for c in columns:
        for t in ["peak", "trough"]:
            plot_df = peaks_summary_df.loc[peaks_summary_df["type"] == t]
            plot_df = plot_df.loc[plot_df["measure"] == c]
            fig = px.parallel_coordinates(plot_df, dimensions=["num_of_peaks", "width", "prominence", "importance"],
                                          color="story_id", color_continuous_scale=px.colors.diverging.Tealrose, )

            export_plots(args, f"/prediction_peaks/multi/{c}_{t}_parallel_summary", fig)

            polar_list = []
            for index, row in plot_df.iterrows():
                print(row)
                for c in ["num_of_peaks","prominence", "width", "importance"]:
                    polar_list.append({"story_id": row["story_id"], "type": row["type"], "measure": row["measure"],"theta": c, "r": row[c]})

            if len(polar_list) == 0:
                continue

            #plot_polar_df = pandas.DataFrame(polar_list)
            #print(plot_polar_df)
            #fig = px.line_polar(plot_polar_df, r="r", theta="theta",
            #                    color="story_id", line_close=True)

            export_plots(args, f"/prediction_peaks/multi/{c}_{t}_polar", fig)

    full_table_list = []
    for m in ["num_of_peaks", "width", "prominence", "importance"]:
        for m2 in ["num_of_peaks", "width", "prominence", "importance"]:

            for t in ["peak","trough"]:
                for t2 in ["peak", "trough"]:

                    table_list = []

                    pear_corr = []
                    ken_corr = []
                    spear_corr = []

                    for c in columns:

                        pear_corr_ann = []
                        ken_corr_ann = []
                        spear_corr_ann = []

                        for c2 in columns:

                            type_x_df = peaks_summary_df.loc[peaks_summary_df["type"] == t]

                            x_df = type_x_df.loc[type_x_df["measure"] == c]
                            x_list = list(x_df[m])

                            type_y_df = peaks_summary_df.loc[peaks_summary_df["type"] == t2]
                            y_df = type_y_df.loc[type_y_df["measure"] == c2]
                            y_list = list(y_df[m2])

                            kendall, pearson, spearman = calculate_correlation(c, c2, table_list, x_list, y_list, measure=m, measure2=m2)

                            pear_corr_ann.append(pearson)
                            spear_corr_ann.append(spearman)
                            ken_corr_ann.append(kendall)

                        pear_corr.append(pear_corr_ann)
                        ken_corr.append(ken_corr_ann)
                        spear_corr.append(spear_corr_ann)

                    full_table_list.extend(table_list)

                    export_correlations(args, f"/prediction_peaks/self_correlation/{m}_{m2}_{t}_{t2}_", columns, columns, ken_corr, pear_corr,
                                        spear_corr,
                                        table_list)
    full_table_df = pandas.DataFrame(full_table_list)
    full_table_df.to_csv(f"{args['output_dir']}/prediction_peaks/self_correlation/all_correlation.csv")

    agg_annotation_df = aggregate_annotations_df(annotation_df)

    annotation_measures = list(agg_annotation_df['name'].unique())
    annotation_measures.sort()

    story_ids = annotation_df[annotation_story_id_column].unique()
    story_ids = [int(s) for s in story_ids]
    story_ids.sort()

    full_table_list = []
    for m in ["num_of_peaks", "width", "prominence", "importance"]:

        for t in ["peak", "trough"]:

            for b in peaks_df['c'].unique():

                table_list = []

                pear_corr = []
                ken_corr = []
                spear_corr = []

                peaks_group_df = peaks_summary_df.loc[peaks_summary_df["c"] == b]

                columns_in_group = list(set(columns).intersection(set(peaks_group_df["measure"].unique())))
                for c in sorted(columns_in_group):

                    pear_corr_ann = []
                    ken_corr_ann = []
                    spear_corr_ann = []

                    prediction_list = []

                    for am in annotation_measures:

                        x_list = []
                        y_list = []

                        for story_id in story_ids:
                            agg_story_row = agg_annotation_df.loc[agg_annotation_df["story_id"] == story_id]
                            peak_story_row = peaks_group_df.loc[peaks_group_df["story_id"] == story_id]

                            if len(agg_story_row) == 0 or len(peak_story_row) == 0:
                                continue

                            pred_dict = {}
                            pred_dict["story_id"] = story_id



                            pred_dict["z"] = am

                            pred_row = agg_story_row.loc[agg_story_row["name"] == am]
                            y = float(pred_row.iloc[0]["mean"])

                            type_x_df = peak_story_row.loc[peak_story_row["type"] == t]

                            x_df = type_x_df.loc[type_x_df["measure"] == c]

                            x = float(x_df[m])

                            pred_dict["x"] = x
                            x_list.append(x)
                            pred_dict["y"] = y
                            y_list.append(y)

                            prediction_list.append(pred_dict)

                        if len(x_list) >= 2 and len(y_list) >= 2:

                            kendall, pearson, spearman = calculate_correlation(c, am, table_list, x_list, y_list,
                                                                               measure="mean", measure2=m)
                            print(c,am, kendall, pearson, spearman)
                        else:
                            kendall = pearson = spearman = 0.0

                        pear_corr_ann.append(pearson)
                        spear_corr_ann.append(spearman)
                        ken_corr_ann.append(kendall)


                    pear_corr.append(pear_corr_ann)
                    ken_corr.append(ken_corr_ann)
                    spear_corr.append(spear_corr_ann)

                    point_df = pandas.DataFrame(data=prediction_list)
                    fig = px.scatter(point_df, x="x", y="y", color="z", trendline="lowess", hover_name="story_id")
                    export_plots(args, f"/prediction_peaks/annotation_correlation/scatter/{b}_{m}_{t}_{c}_", fig)

                full_table_list.extend(table_list)

                export_correlations(args, f"/prediction_peaks/annotation_correlation//heatmap/{b}_{m}_{t}_", annotation_stats_columns,
                                    columns_in_group, ken_corr, pear_corr,
                                    spear_corr,
                                    table_list)

    full_table_df = pandas.DataFrame(full_table_list)
    full_table_df.to_csv(f"{args['output_dir']}/prediction_peaks/annotation_correlation/all_correlation.csv")


def find_and_extract_peaks(story_df, type, x, c):
    story_peak_summary = {}

    max_x = max(x)
    min_x = min(x)

    prom = (max_x - min_x) / 20.0

    peaks, peaks_meta = find_peaks(x, width=1, prominence=prom)
    #print(peaks, peaks_meta)
    peak_list = [dict(zip(peaks_meta, i)) for i in zip(*peaks_meta.values())]
    sentence_ids = story_df["sentence_id"].to_list()

    story_peak_summary["story_id"] = story_df["story_id"].unique()[0]
    story_peak_summary["type"] = type
    story_peak_summary["num_of_peaks"] = len(peak_list)
    story_peak_summary["prominence"] = 0.0
    story_peak_summary["width"] = 0.0
    story_peak_summary["importance"] = 0.0
    story_peak_summary["story_length"] = len(sentence_ids)
    story_peak_summary["measure"] = c

    for index, peak in zip(peaks, peak_list):
        sentence_id = sentence_ids[index]

        row = story_df.loc[story_df["sentence_id"] == sentence_id]

        peak["story_id"] = int(row["story_id"])
        peak["sentence_id"] = int(row["sentence_id"])
        peak["sentence_num"] = int(row["sentence_num"])
        peak["sentence_text"] = row["sentence_text"].values[0]
        peak["type"] = type
        peak["measure"] = c

        story_peak_summary["prominence"] += peak["prominences"]
        story_peak_summary["width"] += peak["widths"]
        story_peak_summary["importance"] += peak["prominences"] * peak["widths"]

    return peak_list, story_peak_summary


def story_stats_correlation(args):
    print(args)

    dfs = []
    for filename in args["annotation_stats"]:
        dfs.append(pandas.read_csv(filename))

    annotation_df = pandas.concat(dfs, ignore_index=True)
    print(annotation_df.columns)

    pred_df = pandas.read_csv(args["batch_stats"])

    story_ids = pred_df["story_id"].to_list()
    story_ids.sort()


    position_df = pandas.read_csv(args["position_stats"])
    print(position_df.columns)

    #prediction_peaks(args, annotation_df, position_df)
    #prediction_position_correlation(args, position_df)
    prediction_correlation(args, pred_df)
    #annotation_correlation(args, annotation_df)
    #prediction_annotation_correlation(args, annotation_df, pred_df)

def prediction_position_correlation(args, position_df):

    ensure_dir(f"{args['output_dir']}/prediction_sentence_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_sentence_correlation/box/")
    ensure_dir(f"{args['output_dir']}/prediction_sentence_correlation/scatter/")

    #print(position_df)
    columns = extract_position_measure_columns(position_df)

    hor_list = []
    for sent_id in position_df["sentence_id"].unique():
        sent_id = int(sent_id)
        sent_df = position_df.loc[position_df["sentence_id"] == sent_id]
        for c in list(columns):
            hor_dict = {}
            hor_dict["sentence_id"] = sent_id
            hor_dict["story_id"] = int(sent_df["story_id"])
            hor_dict["value"] = float(sent_df[c])
            hor_dict["measure"] = c

            hor_list.append(hor_dict)

    point_df = pandas.DataFrame(data=hor_list)
    point_df['c'] = point_df.apply(lambda row: label_bucket(row), axis=1)

    for c in point_df['c'].unique():
        display_df = point_df.loc[point_df['c'] == c]

        fig = px.box(display_df, y="value", x="measure", notched=True)
        export_plots(args, f"/prediction_sentence_correlation/box/{c}_box", fig)

    pear_corr = []
    ken_corr = []
    spear_corr = []

    table_list = []
    for c1 in  columns:

        prediction_list = []

        pear_corr_ann = []
        ken_corr_ann = []
        spear_corr_ann = []

        for c2 in columns:

            if "Unnamed" in c1 or "Unnamed" in c2:
                continue

            #print(c1, c2)
            x_series = list(position_df[c1])
            y_series = list(position_df[c2])
            sentence_ids = list(position_df["sentence_id"])
            sentence_text = list(position_df["sentence_text"])
            x_list = []
            y_list = []

            for s, t, x, y in zip(sentence_ids, sentence_text, x_series, y_series):
                try:
                    x = float(x)
                    y = float(y)
                    x_list.append(x)
                    y_list.append(y)

                    pred_dict = {}

                    pred_dict["z"] = c2
                    pred_dict["x"] = x
                    pred_dict["y"] = y
                    pred_dict["sentence_id"] = s
                    pred_dict["sentence_text"] = t

                    prediction_list.append(pred_dict)

                except:
                    pass

            kendall, pearson, spearman = calculate_correlation(c1, c2, table_list, x_list, y_list)

            pear_corr_ann.append(pearson)
            spear_corr_ann.append(spearman)
            ken_corr_ann.append(kendall)

        pear_corr.append(pear_corr_ann)
        ken_corr.append(ken_corr_ann)
        spear_corr.append(spear_corr_ann)

        point_df = pandas.DataFrame(data=prediction_list)
        fig = px.scatter(point_df, x="x", y="y", title=f"{c1}", color="z", trendline="lowess", hover_name="sentence_text", color_discrete_sequence=px.colors.cyclical.IceFire,)

        export_plots(args, f"/prediction_sentence_correlation/scatter_{c1}", fig)

    export_correlations(args,  "/prediction_sentence_correlation/", columns, columns, ken_corr, pear_corr, spear_corr, table_list)

def export_correlations(args, base, columns, columns2, ken_corr, pear_corr, spear_corr, table_list):

    for corr_type, d in zip(["pearson", "spearman", "kendall"], [pear_corr, spear_corr, ken_corr]):

        print(corr_type, columns, columns2, d)

        # print(measure, d, pred_measures, annotation_stats_columns)

        fig = go.Figure(data=go.Heatmap(
            z=d,
            x=columns,
            y=columns2
        ))

        export_plots(args, f"{base}{corr_type}_heatmap", fig)
    measure_correlation = pandas.DataFrame(table_list)
    measure_correlation.to_csv(f"{args['output_dir']}/{base}_corr.csv")


def calculate_correlation(c1, c2, table_list, x_list, y_list, measure="value", measure2="value"):
    print(c1, c2, x_list, y_list)
    pearson, pearson_p_value = pearsonr(x_list, y_list)
    table_list.append(
        {"c1": c1, "c2": c2, "type": "pearson",
         "correlation": pearson,
         "p_value": pearson_p_value, "measure": measure, "measure_2": measure2})
    kendall, kendall_p_value = kendalltau(x_list, y_list, nan_policy="omit")
    table_list.append(
        {"c1": c1, "c2": c2, "type": "kendall",
         "correlation": kendall,
         "p_value": kendall_p_value, "measure": measure, "measure_2": measure2})
    spearman, spearman_p_value = spearmanr(x_list, y_list)
    table_list.append(
        {"c1": c1, "c2": c2, "type": "spearman",
         "correlation": spearman,
         "p_value": spearman_p_value, "measure": measure, "measure_2": measure2})
    return kendall, pearson, spearman


def export_plots(args, file, fig):
    if not args["no_html_plots"]:
        file_path = f"{args['output_dir']}/{file}.html"
        print(f"Save plot: {file_path}")
        pio.write_html(fig, file_path)
    if not args["no_pdf_plots"]:
        file_path = f"{args['output_dir']}/{file}.pdf"
        print(f"Save plot pdf: {file_path}")
        pio.write_image(fig, file_path)


def extract_position_measure_columns(position_df):
    columns = list(set(list(position_df.columns)).difference(
        {"name", "story_id", "sentence_id", "sentence_text", "steps", "Unnamed: 0"}))
    columns.sort()
    columns = [c for c in columns if
               not c.endswith('_1') and not c.endswith('_2') and not c.endswith('_3') and not c.endswith('_4')]
    return columns


def annotation_correlation(args, annotation_df):

    ensure_dir(f"{args['output_dir']}/annotation_correlation/")

    ensure_dir(f"{args['output_dir']}/annotation_correlation/multi/")
    ensure_dir(f"{args['output_dir']}/annotation_correlation/heatmap/")
    ensure_dir(f"{args['output_dir']}/annotation_correlation/scatter/")

    agg_annotation_df = aggregate_annotations_df(annotation_df)
    annotation_measures = list(agg_annotation_df['name'].unique())
    annotation_measures.sort()

    print(agg_annotation_df)
    print(agg_annotation_df.columns)

    for m in ["mean", "median", "std"]:
        fig = px.line_polar(agg_annotation_df, r="mean", theta="name",
                            color="story_id", line_close=True)
        export_plots(args, f"/annotation_correlation/multi/{m}_polar", fig)

    story_ids = list(agg_annotation_df['story_id'].unique())

    fig = px.box(agg_annotation_df, y="mean", x="name", notched=True)

    export_plots(args, "/annotation_correlation/box", fig)

    pear_corr = []
    ken_corr = []
    spear_corr = []
    table_list = []

    for am in annotation_measures:
        prediction_list = []

        pear_corr_ann = []
        ken_corr_ann = []
        spear_corr_ann = []

        for am2 in annotation_measures:

            x_list = []
            y_list = []

            for story_id in story_ids:
                story_row = agg_annotation_df.loc[agg_annotation_df["story_id"] == story_id]

                pred_dict = {}
                pred_dict["story_id"] = story_id

                #print(am, am2, story_id)

                pred_dict["z"] = am2

                pred_row = story_row.loc[story_row["name"] == am]
                x = float(pred_row.iloc[0]["mean"])
                pred_dict["x"] = x
                x_list.append(x)
                pred_dict["x_err"] = float(pred_row.iloc[0]["std"])

                pred_row = story_row.loc[story_row["name"] == am2]
                y = float(pred_row.iloc[0]["mean"])
                pred_dict["y"] = y
                y_list.append(y)
                pred_dict["y_err"] = float(pred_row.iloc[0]["std"])

                prediction_list.append(pred_dict)

            kendall, pearson, spearman = calculate_correlation(am, am2, table_list, x_list, y_list)

            pear_corr_ann.append(pearson)
            spear_corr_ann.append(spearman)
            ken_corr_ann.append(kendall)

        pear_corr.append(pear_corr_ann)
        ken_corr.append(ken_corr_ann)
        spear_corr.append(spear_corr_ann)

        point_df = pandas.DataFrame(data=prediction_list)
        fig = px.scatter(point_df, x="x", y="y", color="z", title=am, trendline="lowess", hover_name="story_id")

        export_plots(args, f"/annotation_correlation/scatter/{am}", fig)

    export_correlations(args, f"/annotation_correlation/heatmap/", annotation_stats_columns, annotation_stats_columns, ken_corr, pear_corr, spear_corr,
                        table_list)

def prediction_correlation(args,  pred_df):

    ensure_dir(f"{args['output_dir']}/prediction_summary_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_summary_correlation/multi/")
    ensure_dir(f"{args['output_dir']}/prediction_summary_correlation/box/")
    ensure_dir(f"{args['output_dir']}/prediction_summary_correlation/scatter/")
    ensure_dir(f"{args['output_dir']}/prediction_summary_correlation/heatmap/")

    story_ids = pred_df['story_id'].unique()
    pred_measures = pred_df['name'].unique()
    #print(pred_measures)

    pred_df['c'] = pred_df.apply(lambda row: label_bucket(row, attribute="name"), axis=1)

    for c in pred_df["c"].unique():

        display_df = pred_df.loc[pred_df["c"] == c]

        for m in ["mean", "50_perc", "std"]:

            fig = px.box(display_df, y=m, x="name", notched=True)
            export_plots(args, f"/prediction_summary_correlation/box/{c}_{m}", fig)

            fig = px.line_polar(display_df, r=m, theta="name",
                                color="story_id", line_close=True)
            export_plots(args, f"/prediction_summary_correlation/multi/{c}_{m}_polar", fig)


        for pm in pred_measures:
            prediction_list = []
            for story_id in story_ids:
                story_row = pred_df.loc[pred_df["story_id"] == story_id]

                for pm2 in sorted(list(set(pred_measures).intersection(set(display_df["name"].unique())))):
                    pred_dict = {}
                    pred_dict["story_id"] = story_id

                    pred_dict["z"] = pm2

                    pred_row = story_row.loc[story_row["name"] == pm]
                    pred_dict["x"] = float(pred_row.iloc[0]["mean"])
                    pred_dict["x_err"] = float(pred_row.iloc[0]["std"])

                    pred_row = story_row.loc[story_row["name"] == pm2]
                    pred_dict["y"] = float(pred_row.iloc[0]["mean"])
                    pred_dict["y_err"] = float(pred_row.iloc[0]["std"])

                    prediction_list.append(pred_dict)

            point_df = pandas.DataFrame(data=prediction_list)

            fig = px.scatter(point_df, x="x", y="y", color="z", title=pm, trendline="lowess", hover_name="story_id")

            export_plots(args, f"/prediction_summary_correlation/scatter/{c}_{pm}", fig)

    full_table_list = []
    for m in ["mean", "std"]:


        table_list = []

        ken_corr = []
        pear_corr = []
        spear_corr = []

        for pm in pred_measures:

            ken_corr_ann = []
            pear_corr_ann = []
            spear_corr_ann = []

            for pm2 in pred_measures:

                x_list = []
                y_list = []
                for story_id in story_ids:

                    story_row = pred_df.loc[pred_df["story_id"] == story_id]
                    pred_row = story_row.loc[story_row["name"] == pm]
                    x = float(pred_row.iloc[0][m])

                    pred_row = story_row.loc[story_row["name"] == pm2]
                    y = float(pred_row.iloc[0][m])

                    x_list.append(x)
                    y_list.append(y)

                kendall, pearson, spearman = calculate_correlation(pm, pm2, table_list, x_list, y_list)
                print(pm, pm2, kendall, pearson, spearman)

                pear_corr_ann.append(pearson)
                spear_corr_ann.append(spearman)
                ken_corr_ann.append(kendall)

            ken_corr.append(ken_corr_ann)
            pear_corr.append(pear_corr_ann)
            spear_corr.append(spear_corr_ann)

        full_table_list.extend(table_list)

        export_correlations(args, f"/prediction_summary_correlation/heatmap/{m}_", pred_measures,
                            pred_measures, ken_corr, pear_corr, spear_corr,
                            table_list)

    full_table_df = pandas.DataFrame(full_table_list)
    full_table_df.to_csv(f"{args['output_dir']}/prediction_summary_correlation/all_correlation.csv")


def prediction_annotation_correlation(args, annotation_df, pred_df):

    ensure_dir(f"{args['output_dir']}/prediction_annotation_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_annotation_correlation/heatmap/")
    ensure_dir(f"{args['output_dir']}/prediction_annotation_correlation/scatter/")

    pred_df['c'] = pred_df.apply(lambda row: label_bucket(row,attribute="name"), axis=1)

    pred_measures = pred_df['name'].unique()

    agg_annotation_df = aggregate_annotations_df(annotation_df)

    story_ids = annotation_df[annotation_story_id_column].unique()
    story_ids = [int(s) for s in story_ids]
    story_ids.sort()

    full_table_list = []
    for m in ["mean", "std"]:

        ken_corr = []
        pear_corr = []
        spear_corr = []

        for b in pred_df['c'].unique():

            table_list = []

            pred_group_df = pred_df.loc[pred_df["c"] == b]

            pred_measures = list(set(pred_measures).intersection(set(pred_group_df["name"].unique())))
            for pm in pred_measures:

                ken_corr_ann = []
                pear_corr_ann = []
                spear_corr_ann = []

                prediction_list = []

                for ac in annotation_stats_columns:

                    x_list = []
                    y_list = []
                    for story_id in story_ids:

                        pred_dict = {}

                        pred_dict['z'] = ac
                        pred_dict['story_id'] = story_id

                        story_row = pred_group_df.loc[pred_group_df["story_id"] == story_id]
                        story_row_ann = agg_annotation_df.loc[agg_annotation_df["story_id"] == story_id]

                        if len(story_row_ann) > 0 and len(story_row) > 0:

                            pred_row = story_row.loc[story_row["name"] == pm]
                            x = float(pred_row.iloc[0][m])
                            pred_dict["x"] = x

                            pred_row_ann = story_row_ann.loc[story_row_ann["name"] == ac]
                            y = pred_row_ann.iloc[0]["mean"]
                            pred_dict["y"] = y
                            # print(float(pred_row_ann.iloc[0][m]))

                            x_list.append(x)
                            y_list.append(y)

                        prediction_list.append(pred_dict)

                    kendall, pearson, spearman = calculate_correlation(pm, ac, table_list, x_list, y_list, measure=pm, measure2="mean")

                    pear_corr_ann.append(pearson)
                    spear_corr_ann.append(spearman)
                    ken_corr_ann.append(kendall)

                point_df = pandas.DataFrame(data=prediction_list)
                fig = px.scatter(point_df, x="x", y="y", color="z", trendline="lowess", hover_name="story_id")

                export_plots(args, f"/prediction_annotation_correlation/scatter/{b}_{pm}_{m}", fig)

                ken_corr.append(ken_corr_ann)
                pear_corr.append(pear_corr_ann)
                spear_corr.append(spear_corr_ann)

            full_table_list.extend(table_list)

            export_correlations(args, f"/prediction_annotation_correlation/heatmap/{m}_", pred_measures, annotation_stats_columns, ken_corr, pear_corr,
                                spear_corr, table_list)

    full_table_df = pandas.DataFrame(full_table_list)
    full_table_df.to_csv(f"{args['output_dir']}/prediction_annotation_correlation/all_correlation.csv")

def aggregate_annotations_df(annotation_df):
    annotations_list = []
    for col in annotation_stats_columns:
        col_df = annotation_df.groupby([annotation_story_id_column], as_index=False).agg(
            {col: ['mean', 'var', 'std', 'median']})

        #print(col_df)
        col_df.columns = col_df.columns.droplevel(0)

        col_df['name'] = col
        #print(col_df)
        col_df = col_df.rename(columns={col_df.columns[0]: "story_id", col_df.columns[5]: 'name'})

        annotations_list.append(col_df)
    agg_annotation_df = pandas.concat(annotations_list, ignore_index=True)
    return agg_annotation_df

story_stats_correlation(vars(args))
