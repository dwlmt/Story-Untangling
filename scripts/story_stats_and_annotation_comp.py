import argparse
import csv
import os
import random
import re

import pandas
from scipy.stats import kendalltau, pearsonr, spearmanr
from statsmodels.compat import scipy

import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

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

def story_stats_correlation(args):
    print(args)

    ensure_dir(f"{args['output_dir']}/prediction_annotation_correlation/")
    ensure_dir(f"{args['output_dir']}/annotation_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_summary_correlation/")
    ensure_dir(f"{args['output_dir']}/prediction_sentence_correlation/")

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

    prediction_position_correlation(args, position_df)
    prediction_correlation(args, pred_df)
    annotation_correlation(args, annotation_df)
    prediction_annotation_correlation(args, annotation_df, pred_df)

def prediction_position_correlation(args, position_df):
    #print(position_df)
    columns = list(set(list(position_df.columns)).difference(
        {"name", "story_id", "sentence_id", "sentence_text", "steps", "Unnamed: 0"}))
    columns.sort()

    columns = [c for c in columns if not c.endswith('_1') and not c.endswith('_2') and not c.endswith('_3') and not c.endswith('_4')]

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
    fig = px.box(point_df, y="value", x="measure", notched=True)

    if not args["no_html_plots"]:
        file_path = f"{args['output_dir']}/prediction_sentence_correlation/box.html"
        print(f"Save plot: {file_path}")
        pio.write_html(fig, file_path)

    if not args["no_pdf_plots"]:
        file_path = f"{args['output_dir']}/prediction_sentence_correlation/box.pdf"
        print(f"Save plot pdf: {file_path}")
        pio.write_image(fig, file_path)

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

            pearson, pearson_p_value = pearsonr(x_list, y_list)
            table_list.append(
                {"prediction_measure": c1, "annotation_measure": c2, "type": "pearson",
                 "correlation": pearson,
                 "p_value": pearson_p_value, "metric": "value", })
            pear_corr_ann.append(pearson)

            kendall, kendall_p_value = kendalltau(x_list, y_list, nan_policy="omit")
            table_list.append(
                {"prediction_measure": c1, "annotation_measure": c2, "type": "kendall",
                 "correlation": kendall,
                 "p_value": kendall_p_value, "metric": "value", })
            ken_corr_ann.append(kendall)

            spearman, spearman_p_value = spearmanr(x_list, y_list)
            table_list.append(
                {"prediction_measure": c1, "annotation_measure": c2, "type": "spearman",
                 "correlation": spearman,
                 "p_value": spearman_p_value, "metric": "value", })
            spear_corr_ann.append(spearman)

        pear_corr.append(pear_corr_ann)
        ken_corr.append(ken_corr_ann)
        spear_corr.append(spear_corr_ann)

        point_df = pandas.DataFrame(data=prediction_list)
        fig = px.scatter(point_df, x="x", y="y", title=f"{c1}", color="z", trendline="lowess", hover_name="sentence_text", color_discrete_sequence=px.colors.cyclical.IceFire,)

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/prediction_sentence_correlation/scatter_{c1}.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)

        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/prediction_sentence_correlation/scatter_{c1}.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)

    for corr_type, d in zip(["pearson", "spearman", "kendall"], [pear_corr, spear_corr, ken_corr]):
        #print(measure, d, pred_measures, annotation_stats_columns)
        fig = go.Figure(data=go.Heatmap(
            z=d,
            x=columns,
            y=columns
        ))

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/prediction_sentence_correlation/heat_map_{corr_type}.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)

        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/prediction_sentence_correlation/heat_map_{corr_type}.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)
    measure_correlation = pandas.DataFrame(table_list)
    measure_correlation.to_csv(f"{args['output_dir']}/prediction_sentence_correlation/corr.csv")

def annotation_correlation(args, annotation_df):
    agg_annotation_df = aggregate_annotations_df(annotation_df)
    annotation_measures = list(agg_annotation_df['name'].unique())
    story_ids = list(agg_annotation_df['story_id'].unique())

    fig = px.box(agg_annotation_df, y="mean", x="name", notched=True)

    if not args["no_html_plots"]:
        file_path = f"{args['output_dir']}/annotation_correlation/box.html"
        print(f"Save plot: {file_path}")
        pio.write_html(fig, file_path)

    if not args["no_pdf_plots"]:
        file_path = f"{args['output_dir']}/annotation_correlation/box.pdf"
        print(f"Save plot pdf: {file_path}")
        pio.write_image(fig, file_path)

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

            pearson, pearson_p_value = pearsonr(x_list, y_list)
            table_list.append(
                {"prediction_measure": am, "annotation_measure": am2, "type": "pearson",
                 "correlation": pearson,
                 "p_value": pearson_p_value, "metric": "value", })
            pear_corr_ann.append(pearson)

            kendall, kendall_p_value = kendalltau(x_list, y_list, nan_policy="omit")
            table_list.append(
                {"prediction_measure": am, "annotation_measure": am2, "type": "kendall",
                 "correlation": kendall,
                 "p_value": kendall_p_value, "metric": "value", })
            ken_corr_ann.append(kendall)

            spearman, spearman_p_value = spearmanr(x_list, y_list)
            table_list.append(
                {"prediction_measure": am, "annotation_measure": am2, "type": "spearman",
                 "correlation": spearman,
                 "p_value": spearman_p_value, "metric": "value", })
            spear_corr_ann.append(spearman)

        pear_corr.append(pear_corr_ann)
        ken_corr.append(ken_corr_ann)
        spear_corr.append(spear_corr_ann)

        point_df = pandas.DataFrame(data=prediction_list)
        fig = px.scatter(point_df, x="x", y="y", color="z", title=am, trendline="lowess", hover_name="story_id")

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/annotation_correlation/scatter_{am}.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)

        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/annotation_correlation/scatter_{am}.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)

    for corr_type, d in zip(["pearson", "spearman", "kendall"], [pear_corr, spear_corr, ken_corr]):
        #print(measure, d, pred_measures, annotation_stats_columns)
        fig = go.Figure(data=go.Heatmap(
            z=d,
            x=annotation_stats_columns,
            y=annotation_stats_columns
        ))

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/annotation_correlation/heat_map_{corr_type}.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)

        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/annotation_correlation/heat_map_{corr_type}.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)
    measure_correlation = pandas.DataFrame(table_list)
    measure_correlation.to_csv(f"{args['output_dir']}/annotation_correlation/corr.csv")

    print(agg_annotation_df)

def prediction_correlation(args,  pred_df):
    story_ids = pred_df['story_id'].unique()
    pred_measures = pred_df['name'].unique()
    #print(pred_measures)

    fig = px.box(pred_df, y="mean", x="name", notched=True)

    if not args["no_html_plots"]:
        file_path = f"{args['output_dir']}/prediction_summary_correlation/box.html"
        print(f"Save plot: {file_path}")
        pio.write_html(fig, file_path)

    if not args["no_pdf_plots"]:
        file_path = f"{args['output_dir']}/prediction_summary_correlation/box.pdf"
        print(f"Save plot pdf: {file_path}")
        pio.write_image(fig, file_path)

    for pm in pred_measures:
        prediction_list = []
        for story_id in story_ids:
            story_row = pred_df.loc[pred_df["story_id"] == story_id]

            for pm2 in pred_measures:
                pred_dict = {}
                pred_dict["story_id"] = story_id

                print(pm, pm2, story_id)

                pred_dict["z"] = pm2

                pred_row = story_row.loc[story_row["name"] == pm]
                pred_dict["x"] = float(pred_row.iloc[0]["mean"])
                pred_dict["x_err"] = float(pred_row.iloc[0]["std"])

                pred_row = story_row.loc[story_row["name"] == pm2]
                pred_dict["y"] = float(pred_row.iloc[0]["mean"])
                pred_dict["y_err"] = float(pred_row.iloc[0]["std"])

                prediction_list.append(pred_dict)

        point_df = pandas.DataFrame(data=prediction_list)
        #print(point_df)
        #error_x="x_err", error_y="y_err",
        fig = px.scatter(point_df, x="x", y="y", color="z", title=pm, trendline="lowess", color_discrete_sequence=px.colors.cyclical.IceFire, hover_name="story_id")

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/prediction_summary_correlation/scatter_{pm}.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)

        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/prediction_summary_correlation/scatter_{pm}.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)

    table_list = []
    for m in ["mean", "std"]:

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

                pearson, pearson_p_value = pearsonr(x_list, y_list)
                table_list.append(
                    {"prediction_measure": pm, "annotation_measure": pm2, "type": "pearson",
                     "correlation": pearson,
                     "p_value": pearson_p_value, "metric": m, })
                pear_corr_ann.append(pearson)

                kendall, kendall_p_value = kendalltau(x_list, y_list)
                table_list.append(
                    {"prediction_measure": pm, "annotation_measure": pm2, "type": "kendall",
                     "correlation": kendall,
                     "p_value": kendall_p_value, "metric": m, })
                ken_corr_ann.append(kendall)

                spearman, spearman_p_value = spearmanr(x_list, y_list)
                table_list.append(
                    {"prediction_measure": pm, "annotation_measure": pm2, "type": "spearman",
                     "correlation": spearman,
                     "p_value": spearman_p_value, "metric": m, })
                spear_corr_ann.append(spearman)

            ken_corr.append(ken_corr_ann)
            pear_corr.append(pear_corr_ann)
            spear_corr.append(spear_corr_ann)
    for corr_type, d in zip(["pearson", "kendall", "spearman"], [pear_corr, ken_corr, spear_corr]):
        #print(measure, d, pred_measures, annotation_stats_columns)
        fig = go.Figure(data=go.Heatmap(
            z=d,
            x=pred_measures,
            y=pred_measures
        ))

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/prediction_summary_correlation/heat_map_{corr_type}.html"
            print(f"Save plot: {file_path}")
            pio.write_html(fig, file_path)

        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/prediction_summary_correlation/heat_map_{corr_type}.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)
    measure_correlation = pandas.DataFrame(table_list)
    measure_correlation.to_csv(f"{args['output_dir']}/prediction_summary_correlation/corr.csv")
    #print(measure_correlation)

def prediction_annotation_correlation(args, annotation_df, pred_df):

    pred_measures = pred_df['name'].unique()

    agg_annotation_df = aggregate_annotations_df(annotation_df)

    story_ids = annotation_df[annotation_story_id_column].unique()
    story_ids = [int(s) for s in story_ids]
    story_ids.sort()
    print(story_ids)

    #annotation_df = annotation_df.loc[annotation_df[annotation_story_id_column] == 5825]
    #print(annotation_df[[annotation_story_id_column,"Input.text"]])

    table_list = []
    for m in ["mean", "std"]:

        ken_corr = []
        pear_corr = []
        spear_corr = []

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

                    story_row = pred_df.loc[pred_df["story_id"] == story_id]
                    story_row_ann = agg_annotation_df.loc[agg_annotation_df["story_id"] == story_id]

                    if len(story_row_ann) > 0 and len(story_row) > 0:

                        print(story_id)

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

                pearson, pearson_p_value = pearsonr(x_list, y_list)
                table_list.append(
                    {"prediction_measure": pm, "annotation_measure": ac, "type": "pearson", "correlation": pearson,
                     "p_value": pearson_p_value, "metric": m, })
                pear_corr_ann.append(pearson)

                kendall, kendall_p_value = kendalltau(x_list, y_list)
                table_list.append(
                    {"prediction_measure": pm, "annotation_measure": ac, "type": "kendall", "correlation": kendall,
                     "p_value": kendall_p_value, "metric": m, })
                ken_corr_ann.append(kendall)

                spearman, spearman_p_value = spearmanr(x_list, y_list)
                table_list.append(
                    {"prediction_measure": pm, "annotation_measure": ac, "type": "spearman", "correlation": spearman,
                     "p_value": spearman_p_value, "metric": m, })
                spear_corr_ann.append(spearman)

            point_df = pandas.DataFrame(data=prediction_list)
            fig = px.scatter(point_df, x="x", y="y", color="z", trendline="lowess", hover_name="story_id")

            if not args["no_html_plots"]:
                file_path = f"{args['output_dir']}/prediction_annotation_correlation/scatter_{pm}_{m}.html"
                print(f"Save plot: {file_path}")
                pio.write_html(fig, file_path)

            if not args["no_pdf_plots"]:
                file_path = f"{args['output_dir']}/prediction_annotation_correlation/scatter_{pm}_{m}.pdf"
                print(f"Save plot pdf: {file_path}")
                pio.write_image(fig, file_path)

            ken_corr.append(ken_corr_ann)
            pear_corr.append(pear_corr_ann)
            spear_corr.append(spear_corr_ann)

        for measure, d in zip(["pearson", "kendall", "spearman"], [pear_corr, ken_corr, spear_corr]):
            print(measure, d, pred_measures, annotation_stats_columns)
            fig = go.Figure(data=go.Heatmap(
                z=d,
                y=pred_measures,
                x=annotation_stats_columns
            ))

            if not args["no_html_plots"]:
                file_path = f"{args['output_dir']}/prediction_annotation_correlation/heat_map_{measure}_{m}.html"
                print(f"Save plot: {file_path}")
                pio.write_html(fig, file_path)

            if not args["no_pdf_plots"]:
                file_path = f"{args['output_dir']}/prediction_annotation_correlation/heat_map_{measure}_{m}.pdf"
                print(f"Save plot pdf: {file_path}")
                pio.write_image(fig, file_path)
        measure_correlation = pandas.DataFrame(table_list)
        measure_correlation.to_csv(f"{args['output_dir']}/prediction_annotation_correlation/corr_{m}.csv")
        print(measure_correlation)


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
