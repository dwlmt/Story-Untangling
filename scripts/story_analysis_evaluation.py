''' Create Analysis Charts for Stories in bulk based on the preidction output and cluster analysis.

'''
import argparse
import collections
import os
from collections import OrderedDict

import mord
import numpy
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

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

annotator_prediction_column = "suspense"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def ordinal_regression_bucketed_evaluation(merged_df, args):
    merged_df = merged_df.sort_values(by=["worker_id", "story_id", "sentence_num"]).reset_index()

    merged_df = pd.concat([merged_df, merged_df[1:].reset_index(drop=True).add_suffix("_later")],
                          axis=1)
    merged_df = merged_df.loc[merged_df["story_id"] == merged_df["story_id_later"]]

    for col in model_prediction_columns:
        merged_df[f"{col}_diff"] = merged_df[f"{col}_later"] - merged_df[col]

    print(f"Merged rows: {len(merged_df)}")
    # Remove the first line as cannot be judged relatively.
    merged_df = merged_df.loc[merged_df["suspense"] != 0.0]
    max_df = merged_df.groupby(by=["story_id"], as_index=False)["sentence_num"].max()

    df_all = merged_df.merge(max_df, on=['story_id', 'sentence_num'],
                             how='left', indicator=True)

    merged_df = df_all.loc[df_all['_merge'] == "left_only"]

    train_df = merged_df.loc[merged_df['worker_id'] != 'median']
    test_df = merged_df.loc[merged_df['worker_id'] == 'median']

    print(f"Evaluated rows - training {len(train_df)}, test {len(test_df)}")

    results_data = []

    for col in model_prediction_columns:
        for feature_col in [f"{col}_diff", f"{col}_later"]:
            results_dict = OrderedDict()
            results_dict["feature"] = feature_col

            train_features = features(feature_col, train_df)
            train_target = train_df[annotator_prediction_column].astype(int).to_numpy()

            class_weights = class_weight.compute_class_weight('balanced',
                                                              numpy.unique(train_target),
                                                              train_target)

            sample_weights = [class_weights[x - 1] for x in train_target]

            print("Class Weights", class_weights)

            test_features = features(feature_col, test_df)
            test_target = test_df[annotator_prediction_column].astype(int).to_numpy()

            model = mord.LogisticIT(alpha=0.0)

            params = {}

            pipeline = Pipeline([  # ('column', StandardScaler()),
                ('model', model)])

            print('Estimator: ', model)
            grid = GridSearchCV(pipeline, params,
                                scoring='neg_mean_absolute_error',
                                n_jobs=1, cv=args["folds"])
            grid.fit(train_features, train_target, model__sample_weight=sample_weights)
            pred = grid.best_estimator_.predict(train_features)
            classification_report = metrics.classification_report(train_target, numpy.round(pred).astype(int),
                                                                  output_dict=True)
            results_dict["train_results"] = classification_report

            results_dict["test_results"] = classification_report

            pred = grid.best_estimator_.predict(test_features)
            classification_report = metrics.classification_report(test_target, numpy.round(pred).astype(int),
                                                                  output_dict=True)

            print(classification_report)

            results_dict["test_results"] = classification_report

            results_data.append(flatten(results_dict))

    results_df = pd.DataFrame(data=results_data)
    print(results_df)

    results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/results.csv")


def features(feature_col, train_df):
    train_features = train_df[feature_col].to_numpy()
    train_features = train_features.reshape(-1, 1)
    return train_features


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
