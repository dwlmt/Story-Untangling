''' Create Analysis Charts for Stories in bulk based on the preidction output and cluster analysis.

'''
import argparse
import os

import mord
import numpy
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--position-stats', required=True, type=str, help="CSV of the prediction position stats.")
parser.add_argument('--vector-stats', required=False, type=str, help="CSV containing the vector output.")
parser.add_argument('--annotator-targets', required=True, type=str, help="CSV with consensus predictions.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument("--no-html-plots", default=False, action="store_true", help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true", help="Don't save plots to PDF")
parser.add_argument("--folds", default=5, type=int, help="Folds in the cross validation.")

args = parser.parse_args()

model_prediction_columns = ["generated_surprise_word_overlap",
                            "generated_surprise_simple_embedding",
                            'generated_surprise_l1', 'generated_surprise_l2',
                            'generated_suspense_l1', 'generated_suspense_l2',
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
                            'corpus_suspense_l1_state', 'corpus_suspense_l2_state']

annotator_prediction_column = "median_judgement"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


def ordinal_regression_bucketed_evaluation(merged_df, args):
    merged_df = merged_df.sort_values(by=["story_id", "sentence_num"]).reset_index()

    merged_df = pd.concat([merged_df, merged_df[1:].reset_index(drop=True).add_suffix("_later")],
                          axis=1)
    merged_df = merged_df.loc[merged_df["story_id"] == merged_df["story_id_later"]]

    for col in model_prediction_columns:
        merged_df[f"{col}_diff"] = merged_df[f"{col}_later"] - merged_df[col]

    print(f"Merged rows: {len(merged_df)}")
    # Remove the first line as cannot be judged relatively.
    merged_df = merged_df.loc[merged_df["median_judgement"] != 0.0]
    max_df = merged_df.groupby(by=["story_id"], as_index=False)["sentence_num"].max()

    df_all = merged_df.merge(max_df, on=['story_id', 'sentence_num'],
                             how='left', indicator=True)

    merged_df = df_all.loc[df_all['_merge'] == "left_only"]
    print(merged_df.columns)

    results_data = []

    print(f"Evaluated rows: {len(merged_df)}")

    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

    def acc_fun(target_true, target_fit):
        target_fit = numpy.round(target_fit)
        target_fit.astype('int')
        return accuracy_score(target_true, target_fit)

    def prec_fun(target_true, target_fit):
        target_fit = numpy.round(target_fit)
        target_fit.astype('int')
        return precision_score(target_true, target_fit, average="macro")

    def recall_fun(target_true, target_fit):
        target_fit = numpy.round(target_fit)
        target_fit.astype('int')
        return recall_score(target_true, target_fit, average="macro")

    def f1_fun(target_true, target_fit):
        target_fit = numpy.round(target_fit)
        target_fit.astype('int')
        return f1_score(target_true, target_fit, average="macro")

    for col in model_prediction_columns:
        for feature_col in [f"{col}_diff", f"{col}"]:

            results_dict = {}
            results_dict["feature"] = feature_col

            features = merged_df[feature_col].to_numpy()
            features = features.reshape(-1, 1)

            # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            # features = scaler.fit_transform(features)

            target = merged_df[annotator_prediction_column].astype(int).to_numpy()

            scoring = {'accuracy': make_scorer(acc_fun),
                       'precision': make_scorer(prec_fun),
                       'recall': make_scorer(recall_fun),
                       'f1_score': make_scorer(f1_fun),
                       'mean_absolute_error': make_scorer(mean_absolute_error)
                       }

            model = mord.LAD(max_iter=10000)
            model.fit(features, target)

            res = cross_validate(model,
                                 features,
                                 target,
                                 cv=args["folds"],
                                 scoring=scoring,
                                 return_train_score=True,
                                 )
            print(feature_col, res)

            for key in set(res.keys()).difference({"estimator"}):
                results_dict[f"{key}"] = numpy.mean(res[key])
                results_dict[f"{key}_big_decrease"], results_dict[f"{key}_decrease"], results_dict[f"{key}_same"], \
                results_dict[f"{key}_increase"], results_dict[f"{key}_big_increase"] = res[key]

            results_data.append(results_dict)

    results_df = pd.DataFrame(data=results_data)
    print(results_df)

    results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/results.csv")


def evaluate_stories(args):
    print(f"Evaluate stories: {args}")

    ensure_dir(f"{args['output_dir']}/sentence_model_evaluation/")

    position_df = pd.read_csv(args["position_stats"])
    print(f"Position rows : {len(position_df)}")
    annotator_df = pd.read_csv(args["annotator_targets"])
    print(f"Annotator rows: {len(annotator_df)}")

    vector_df = None
    if "vector_stats" in args and len(args["vector_stats"]) > 0:
        vector_df = pd.read_csv(args["vector_stats"])

    merged_df = pd.merge(position_df, annotator_df, left_on=["story_id", "sentence_num"],
                         right_on=["story_id", "sentence_num"], how="inner")

    ordinal_regression_bucketed_evaluation(merged_df, args)


evaluate_stories(vars(args))
