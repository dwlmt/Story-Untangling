''' Create Analysis Charts for Stories in bulk based on the prediction output and cluster analysis.

'''
import collections
from collections import OrderedDict
from itertools import combinations

import argparse
import mord
import more_itertools
import numpy
import os
import pandas
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import torch
from nltk import interval_distance, AnnotationTask
from scipy.spatial import distance
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from statsmodels.tsa.stattools import coint, ccf

parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--position-stats', required=True, type=str, help="CSV of the prediction position stats.")
parser.add_argument('--vector-stats', required=False, type=str, help="CSV containing the vector output.")
parser.add_argument('--annotator-targets', required=True, type=str, help="CSV with consensus predictions.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument("--no-html-plots", default=False, action="store_true", help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true", help="Don't save plots to PDF")
parser.add_argument("--folds", default=5, type=int, help="Folds in the cross validation.")
parser.add_argument("--epochs", default=150, type=int, help="Number of Epochs for model fitting.")
parser.add_argument("--exclude-worker-ids", type=str, nargs="*", help="Workers to exclude form the annotations.")
parser.add_argument("--cuda-device", default=0, type=int, help="The default CUDA device.")

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


def ordinal_regression_bucketed_evaluation(annotator_df, position_df, args):
    train_df = prepare_dataset(annotator_df, position_df, keep_first_sentence=False)

    agreement_data = []

    for i, row in train_df.iterrows():
        agreement_data.append(
            {"worker_id": str(row["worker_id"]), "sentence_id": str(row["sentence_id_y"]), "value": row["suspense"],
             "type": "human"})

    print(f"Evaluated rows - training {len(train_df)}, test {len(train_df)}")

    results_data = []

    for col in model_prediction_columns:
        for feature_col in [f"{col}_diff", f"{col}"]:
            results_dict = OrderedDict()
            results_dict["measure"] = feature_col

            train_features = features(feature_col, train_df)
            train_target = train_df[annotator_prediction_column].astype(int).to_numpy()

            class_weights = class_weight.compute_class_weight('balanced', numpy.unique(train_target),
                                                              train_target)

            # class_weights = [max(0.5, min(c, 10.0)) for c in class_weights]

            sample_weights = [class_weights[x - 1] for x in train_target]

            print("Class Weights", class_weights)

            model = mord.LogisticIT(alpha=0.0)

            params = {}

            pipeline = Pipeline([('model', model)])

            print('Estimator: ', model)
            grid = GridSearchCV(pipeline, params,
                                scoring='neg_mean_absolute_error',
                                n_jobs=1, cv=args["folds"])
            grid.fit(train_features, train_target) #model__sample_weight=sample_weights)
            pred = grid.best_estimator_.predict(train_features)
            classification_report = metrics.classification_report(train_target, numpy.round(pred).astype(int),
                                                                  output_dict=True)

            classification_report = flatten(classification_report)

            results_dict = {**results_dict, **classification_report}

            agreement_triples = []
            for pred, target_value, sentence in zip(pred, train_target, train_df["sentence_id_x"]):

                agreement_triples.append((str("model"), str(array_to_first_value(sentence)), pred))
                agreement_triples.append((str("target"), str(array_to_first_value(sentence)), array_to_first_value(target_value)))

                agreement_data.append({"worker_id": f"{feature_col}", "sentence_id": str(sentence),
                                       "value": pred, "type": "model_fitted"})

            agreement(agreement_triples, "regression", results_dict)

            results_data.append(flatten(results_dict))

            proportion_counts =  train_df[f"{feature_col}_scaled"].loc[train_df[f"{feature_col}_scaled"] != 0].value_counts(normalize=True, sort=False)

            total = 0.0
            category_threshold_dict = OrderedDict()
            features_as_numpy = train_df[f"{feature_col}_scaled"].values
            for item, value in proportion_counts.iteritems():
                total += value
                category_threshold_dict[item] = numpy.percentile(features_as_numpy, total * 100)

            for k in ["prop","std"]:

                agreement_triples = []
                results_dict = OrderedDict()
                results_dict["measure"] = f"{feature_col}_{k}"
                predictions = []
                for pred, target_value, sentence in zip(train_features, train_target, train_df["sentence_id_x"]):

                    if k == "std:":

                        if pred >= 2.0:
                            mapped_pred = 5
                        elif pred >= 1.0:
                            mapped_pred = 4
                        elif pred < -2.0:
                            mapped_pred = 1
                        elif pred <= -1.0:
                            mapped_pred = 2
                        else:
                            mapped_pred = 3

                    else:
                        mapped_pred = 5 # Default to the biggest, reassign if less
                        for key, value in category_threshold_dict.items():
                            if pred < value:
                                mapped_pred = key
                                break

                    predictions.append(mapped_pred)

                    agreement_triples.append((str("model"), str(array_to_first_value(sentence)), array_to_first_value(mapped_pred)))
                    agreement_triples.append((str("target"), str(array_to_first_value(sentence)), array_to_first_value(target_value)))

                    agreement_data.append({"worker_id": f"{feature_col}_{k}", "sentence_id": str(sentence),
                                           "value": pred, "type": f"model_{k}"})

                classification_report = metrics.classification_report(train_target, numpy.array(predictions).astype(int),
                                                                      output_dict=True)

                classification_report = flatten(classification_report)

                results_dict = {**results_dict, **classification_report}

                agreement(agreement_triples, "regression", results_dict)

                results_data.append(flatten(results_dict))

    results_df = pd.DataFrame(data=results_data)
    results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/categorical_evaluation.csv")

    agreement_df = pandas.DataFrame(data=agreement_data)
    worker_pairwise_agreements = []
    for (worker, other_worker) in combinations(agreement_df["worker_id"].unique(), 2):
        worker_df = agreement_df.loc[agreement_df["worker_id"] == worker]

        other_worker_df = agreement_df.loc[agreement_df["worker_id"] == other_worker]

        triples = []

        agreement_dict = {}
        agreement_dict["worker_id"] = worker
        agreement_dict["type"] = worker_df["type"].values[0]

        agreement_dict["worker_id_2"] = other_worker
        agreement_dict["type_2"] = other_worker_df["type"].values[0]

        combined_df = pandas.merge(worker_df, other_worker_df, on="sentence_id", how="inner")

        if len(combined_df) > 0:

            predictions = []
            targets = []

            for i, row in combined_df.iterrows():
                triples.append(("worker", str(array_to_first_value(row["sentence_id"])), array_to_first_value(row["value_x"])))
                targets.append(array_to_first_value(row["value_x"]))

                triples.append(("other", str(array_to_first_value(row["sentence_id"])), array_to_first_value(row["value_y"])))
                predictions.append( array_to_first_value(row["value_y"]))

            classification_report = metrics.classification_report(numpy.array(targets).astype(int), numpy.array(predictions).astype(int),
                                                                  output_dict=True)
            classification_report = flatten(classification_report)
            agreement_dict = {**agreement_dict, **classification_report}

            agreement_dict["num_prediction_points"] = len(combined_df)

            agreement(triples, "agreement", agreement_dict)
            worker_pairwise_agreements.append(agreement_dict)

    cross_pairwise_agreements_df = pd.DataFrame(data=worker_pairwise_agreements)
    cross_pairwise_agreements_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/all_pairwise_agreements.csv")

def array_to_first_value(target_value):
    changed_value = target_value
    if isinstance(target_value, (numpy.ndarray, numpy.generic,  torch.Tensor, pandas.Series)):
        changed_value = target_value.tolist()

    if isinstance(changed_value, list):
        changed_value = changed_value[0]

    return changed_value


def agreement(agreement_triples, m, results_dict):
    if len(agreement_triples) > 2 and len(set([t[0] for t in agreement_triples])) > 1:
        t = AnnotationTask(data=agreement_triples, distance=interval_distance)
        results_dict[f"{m}_alpha"] = t.alpha()
        results_dict[f"{m}_agreement"] = t.avg_Ao()


def features(feature_col, train_df):
    train_features = train_df[f"{feature_col}_scaled"].to_numpy()
    train_features = train_features.reshape(-1, 1)
    return train_features


class RelativeToAbsoluteModel(torch.nn.Module):

    def __init__(self, origin_weight=0.0, big_decrease=-0.2, decrease=-0.1, same=0.0, increase=0.1, big_increase=0.2,
                 epsilon=0.01):
        super(RelativeToAbsoluteModel, self).__init__()
        self.origin = torch.nn.Linear(1, 1, bias=False)
        self.big_decrease = torch.nn.Linear(1, 1, bias=False)
        self.decrease = torch.nn.Linear(1, 1, bias=False)
        self.same = torch.nn.Linear(1, 1, bias=False)
        self.increase = torch.nn.Linear(1, 1, bias=False)
        self.big_increase = torch.nn.Linear(1, 1, bias=False)

        # Set the default weights. Grads can be turned off to run these
        torch.nn.init.constant_(self.origin.weight, origin_weight)
        torch.nn.init.constant_(self.big_decrease.weight, big_decrease)
        torch.nn.init.constant_(self.same.weight, same)
        torch.nn.init.constant_(self.decrease.weight, decrease)
        torch.nn.init.constant_(self.increase.weight, increase)
        torch.nn.init.constant_(self.big_increase.weight, big_increase)

        self.epsilon = epsilon

    def forward(self, annotation_series):
        initial = self.origin(torch.tensor([1.0]))

        res_tensor = torch.tensor(initial)

        for cat in annotation_series:

            last = res_tensor[-1]

            cat = cat.item()

            if cat == 1:
                change_value = self.big_decrease(torch.tensor([1.0])).clamp(
                    max=min(0.0, self.decrease(torch.tensor([1.0])).item()) - self.epsilon)
            elif cat == 2:
                change_value = self.decrease(torch.tensor([1.0])).clamp(
                    min=self.big_decrease(torch.tensor([1.0])).item() + self.epsilon, max=0.0 - self.epsilon)
            elif cat == 3:
                change_value = self.same(torch.tensor([1.0])).clamp(
                    min=self.decrease(torch.tensor([1.0])).item() + self.epsilon,
                    max=self.increase(torch.tensor([1.0])).item() - self.epsilon)
            elif cat == 4:
                change_value = self.increase(torch.tensor([1.0])).clamp(min=0.0 + self.epsilon, max=self.big_increase(
                    torch.tensor([1.0])).item() - self.epsilon)
            elif cat == 5:
                change_value = self.big_increase(torch.tensor([1.0])).clamp(
                    min=max(0.0, self.increase(torch.tensor([1.0])).item()) + self.epsilon)

            if cat != 0:
                new_value = last + change_value
                res_tensor = torch.cat((res_tensor, new_value))

        return res_tensor


def contineous_evaluation(annotator_df, position_df, args):
    ''' Maps the relative judgements from the annotations to an absolute scale and
    '''
    train_df = prepare_dataset(annotator_df, position_df, keep_first_sentence=True)

    train_df = train_df.loc[train_df['worker_id'] != 'mean']

    cont_model_predictions(args, train_df)
    cont_model_pred_to_ann(args, train_df)
    cont_worker_to_worker(args, train_df)


def cont_worker_to_worker(args, train_df):
    story_ids = train_df["story_id"].unique()
    worker_story_data = []
    worker_results_data = []
    with torch.cuda.device(args["cuda_device"]):
        base_model = RelativeToAbsoluteModel()
        comparison_one_all = []
        comparison_two_all = []
        story_meta = []

        results_dict = OrderedDict()
        results_dict["measure"] = "worker"
        results_dict["training"] = "fitted"

        for story_id in story_ids:

            story_df = train_df.loc[train_df["story_id"] == story_id]
            worker_ids = story_df["worker_id"].unique()
            for worker_id, worker_id_2 in combinations(worker_ids, 2):
                with torch.no_grad():

                    meta_dict = {}

                    worker_df = story_df.loc[story_df["worker_id"] == worker_id]

                    worker_df_2 = story_df.loc[story_df["worker_id"] == worker_id_2]

                    if len(worker_df) > 0 and len(worker_df_2) > 0:

                        # meta_dict["dataset"] = "train"
                        meta_dict["worker_id"] = worker_id
                        meta_dict["worker_id_2"] = worker_id_2

                        meta_dict["training"] = "fixed"

                        suspense = torch.tensor(worker_df["suspense"].tolist()).int()
                        abs_suspense = base_model(suspense)

                        suspense_2 = torch.tensor(worker_df_2["suspense"].tolist()).int()
                        abs_suspense_2 = base_model(suspense_2)

                        if len(abs_suspense) == 0 or len(abs_suspense_2) == 0:
                            continue

                        comparison_one_all.append(abs_suspense.tolist())
                        comparison_two_all.append(abs_suspense_2.tolist())
                        story_meta.append(meta_dict)

            for story_meta_dict, predictions, annotations in zip(story_meta, comparison_one_all, comparison_two_all):
                abs_evaluate_predictions(predictions, annotations, story_meta_dict)
                worker_story_data.append(story_meta_dict)

            abs_evaluate_predictions(comparison_one_all, comparison_two_all, results_dict)

            worker_results_data.append(results_dict)

        worker_results_df = pandas.DataFrame(data=worker_results_data)
        worker_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/worker_rel_to_abs.csv")

        worker_story_results_df = pandas.DataFrame(data=worker_story_data)
        worker_story_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/worker_rel_to_abs_story.csv")


def cont_model_pred_to_ann(args, train_df):
    results_data = []
    story_data = []

    story_ids = train_df["story_id"].unique()

    with torch.cuda.device(args["cuda_device"]):

        for col in model_prediction_columns:

            fitting_model = RelativeToAbsoluteModel()

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(fitting_model.parameters(), lr=0.01)

            for epoch in range(args["epochs"]):

                results_dict = OrderedDict()
                results_dict["measure"] = col
                results_dict["training"] = "fitted"
                # results_dict["dataset"] = "train"

                comparison_one_all = []
                comparison_two_all = []
                story_meta = []

                story_ids = train_df["story_id"].unique()

                for story_id in story_ids:

                    story_df = train_df.loc[train_df["story_id"] == story_id]
                    worker_ids = story_df["worker_id"].unique()
                    for worker_id in worker_ids:

                        meta_dict = {}

                        worker_df = story_df.loc[story_df["worker_id"] == worker_id]

                        if len(worker_df) > 0:

                            # meta_dict["dataset"] = "train"
                            meta_dict["story_id"] = story_id
                            meta_dict["worker_id"] = worker_id
                            meta_dict["measure"] = col

                            if epoch == 0:
                                meta_dict["training"] = "fixed"
                            else:
                                meta_dict["training"] = "fitted"

                            suspense = torch.tensor(worker_df["suspense"].tolist()).int()
                            model_predictions = torch.tensor(worker_df[f"{col}_scaled"].tolist())

                            abs_suspense = fitting_model(suspense)

                            if len(model_predictions) == 0 or len(abs_suspense) == 0:
                                continue

                            comparison_one_all.append(abs_suspense.tolist())
                            comparison_two_all.append(model_predictions.tolist())
                            story_meta.append(meta_dict)

                            if abs_suspense.size(0) == model_predictions.size(0):
                                loss = criterion(abs_suspense, model_predictions)

                                if epoch > 0:
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                if epoch == 0 or epoch == args["epochs"] - 1:
                    if epoch == 0:
                        results_dict["training"] = "fixed"

                    for story_meta_dict, predictions, annotations in zip(story_meta, comparison_one_all,
                                                                         comparison_two_all):
                        abs_evaluate_predictions(predictions, annotations, story_meta_dict)
                        story_data.append(story_meta_dict)

                    abs_evaluate_predictions(comparison_one_all, comparison_two_all, results_dict)

                    results_data.append(results_dict)
        results_df = pandas.DataFrame(data=results_data)
        results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/model_to_ann_rel_to_abs.csv")

        story_results_df = pandas.DataFrame(data=story_data)
        story_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/model_to_ann_rel_to_abs_story.csv")
    return col, story_ids


def cont_model_predictions(args, train_df):
    prediction_story_data = []
    prediction_results_data = []
    for col, col_2 in combinations(model_prediction_columns, 2):

        comparison_one_all = []
        comparison_two_all = []
        story_meta = []

        results_dict = OrderedDict()
        results_dict["training"] = "fixed"
        results_dict["measure"] = col
        results_dict["measure_2"] = col_2

        story_ids = train_df["story_id"].unique()

        for story_id in story_ids:

            story_df = train_df.loc[train_df["story_id"] == story_id]

            meta_dict = {}

            meta_dict["story_id"] = story_id
            meta_dict["measure"] = col
            meta_dict["measure_2"] = col_2
            meta_dict["training"] = "fixed"

            model_predictions = story_df[f"{col}_scaled"].tolist()
            model_predictions_2 = story_df[f"{col_2}_scaled"].tolist()

            if len(model_predictions) == 0 or len(model_predictions_2) == 0:
                continue

            comparison_one_all.append(model_predictions)
            comparison_two_all.append(model_predictions_2)

            story_meta.append(meta_dict)

        for story_meta_dict, predictions, annotations in zip(story_meta, comparison_one_all, comparison_two_all):
            abs_evaluate_predictions(predictions, annotations, story_meta_dict)
            prediction_story_data.append(story_meta_dict)

        abs_evaluate_predictions(comparison_one_all, comparison_two_all, results_dict)

        prediction_results_data.append(results_dict)
    prediction_results_df = pandas.DataFrame(data=prediction_results_data)
    prediction_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/prediction_rel_to_abs.csv")
    prediction_story_results_df = pandas.DataFrame(data=prediction_story_data)
    prediction_story_results_df.to_csv(
        f"{args['output_dir']}/sentence_model_evaluation/prediction_rel_to_abs_story.csv")


def abs_evaluate_predictions(annotations, predictions, results_dict):
    try:
        if any(isinstance(el, list) for el in annotations):

            coint_pred_to_ann = []
            coint_ann_to_pred = []
            cross_correlation_list = []
            kendall_list = []
            spearman_list = []
            pearson_list = []
            l1_distance_list = []
            l2_distance_list = []

            for pred, ann in zip(predictions, annotations):
                coint_t, coint_t_p_value, coint_t_critical_values = coint(numpy.asarray(ann),
                                                                          numpy.asarray(pred), autolag=None)
                coint_ann_to_pred.append(coint_t)

                coint_t, coint_t_p_value, coint_t_critical_values = coint(numpy.asarray(pred),
                                                                          numpy.asarray(ann), autolag=None)
                coint_pred_to_ann.append(coint_t)

                cross_correlation = ccf(numpy.asarray(pred), numpy.asarray(ann))
                cross_correlation_list.append(numpy.mean(cross_correlation))

                pearson, _ = pearsonr(pred, ann)
                pearson_list.append(pearson)
                kendall, _ = kendalltau(pred, ann, nan_policy="omit")
                kendall_list.append(kendall)
                spearman, _ = spearmanr(pred, ann)
                spearman_list.append(spearman)

                l2_distance_list.append(distance.euclidean(pred, ann))
                l1_distance_list.append(distance.cityblock(pred, ann))

            results_dict[f"first_second_cointegration"] = sum(coint_ann_to_pred) / float(len(annotations))
            results_dict[f"second_first_cointegration"] = sum(coint_pred_to_ann) / float(len(annotations))
            results_dict[f"cross_correlation"] = sum(cross_correlation_list) / float(len(annotations))
            results_dict[f"pearson"] = sum(pearson_list) / float(len(annotations))
            results_dict[f"kendall"] = sum(kendall_list) / float(len(annotations))
            results_dict[f"spearman"] = sum(spearman_list) / float(len(annotations))
            results_dict[f"l2_distance"] = sum(l2_distance_list) / float(len(annotations))
            results_dict[f"l1_distance"] = sum(l1_distance_list) / float(len(annotations))

        else:

            coint_t, coint_t_p_value, coint_t_critical_values = coint(numpy.asarray(annotations),
                                                                      numpy.asarray(predictions),
                                                                      autolag=None)
            results_dict[f"ann_to_pred_cointegration"] = coint_t
            results_dict[f"ann_to_pred_cointegration_p_value"] = coint_t_p_value
            results_dict[f"ann_to_pred_cointegration_critical_1"], results_dict[
                f"ann_to_pred_cointegration_critical_5"], \
            results_dict[f"ann_to_pred_cointegration_critical_10"] = coint_t_critical_values

            coint_t, coint_t_p_value, coint_t_critical_values = coint(numpy.asarray(predictions),
                                                                      numpy.asarray(annotations),
                                                                      autolag=None)
            results_dict[f"pred_to_ann_cointegration"] = coint_t
            results_dict[f"pred_to_ann_cointegration_p_value"] = coint_t_p_value
            results_dict[f"pred_to_ann_cointegration_critical_1"], results_dict[
                f"pred_to_ann_cointegration_critical_5"], results_dict[
                f"pred_to_ann_cointegration_critical_10"] = coint_t_critical_values

            cross_correlation = ccf(numpy.asarray(predictions), numpy.asarray(annotations))
            results_dict[f"cross_correlation"] = numpy.mean(cross_correlation)

            results_dict[f"pearson"], results_dict[f"pearson_p_value"] = pearsonr(predictions,
                                                                                  annotations)
            results_dict[f"kendall"], results_dict[f"kendall_p_value"] = kendalltau(
                predictions, annotations, nan_policy="omit")
            results_dict[f"spearman"], results_dict[f"spearman_p_value"] = spearmanr(
                predictions, annotations)

            results_dict["l2_distance"] = distance.euclidean(predictions, annotations)
            results_dict["l1_distance"] = distance.cityblock(predictions, annotations)

    except Exception as ex:
        print(ex)


def plot_annotator_and_model_predictions(position_df, merged_sentence_df, args, model=RelativeToAbsoluteModel()):
    print(f"Plot the annotator sentences to get a visualisation of the peaks in the annotations.")

    position_df = scale_prediction_columns(position_df)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    story_ids = merged_sentence_df["story_id"].unique()

    position_story_ids = position_df["story_id"].unique()

    story_ids = set(story_ids).intersection(set(position_story_ids))

    with torch.no_grad():

        for story_id in story_ids:

            position_story_df = position_df.loc[position_df["story_id"] == story_id]
            sentence_nums = position_story_df["sentence_num"].tolist()
            sentence_text = position_story_df["sentence_text"].tolist()
            text = sentence_text  # [f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in range(len(sentence_nums))],

            plot_data = []

            story_df = merged_sentence_df.loc[merged_sentence_df["story_id"] == story_id]
            story_df = story_df.groupby(['story_id', 'sentence_id', 'sentence_num', 'worker_id'],
                                        as_index=False).first()

            if len(story_df) > 0:

                worker_ids = set(story_df["worker_id"].unique())

                for worker_id in worker_ids:

                    if worker_id == "mean":
                        continue

                    worker_df = story_df.loc[story_df["worker_id"] == worker_id]

                    sel_col_df = worker_df[['story_id', 'sentence_id', 'sentence_num', 'worker_id', 'suspense']]

                    if len(worker_df) > 0:

                        worker_df = worker_df.sort_values(by=["sentence_num"])

                        suspense = torch.tensor(worker_df["suspense"].tolist()).int()

                        measure_values = model(suspense).tolist()

                        dash = "solid"
                        if worker_id == "median":
                            dash = "dash"

                        trace = go.Scatter(
                            x=worker_df["sentence_num"],
                            y=measure_values,
                            mode='lines+markers',
                            name=f"{worker_id}",
                            text=text,
                            line=dict(color=colors[0], dash=dash)
                        )
                        plot_data.append(trace)

                plot_data.append(trace)

            story_df = position_df.loc[position_df["story_id"] == story_id]

            if len(story_df) > 0:

                for col in model_prediction_columns:

                    dash = "solid"
                    if "corpus" in col:
                        dash = "dash"

                    measure_values = torch.tensor(position_df[f"{col}_scaled"].tolist())

                    measure_offset = 0.0 - measure_values[0]
                    measure_values = [m + measure_offset for m in measure_values]

                    color_lookup = 1
                    if "suspense" in col:
                        color_lookup += 5

                    if "entropy" in col:
                        color_lookup += 1
                    elif "l1" in col:
                        color_lookup += 2
                    elif "l2" in col:
                        color_lookup += 3

                    trace = go.Scatter(
                        x=worker_df["sentence_num"],
                        y=measure_values,
                        mode='lines+markers',
                        name=f"{col}",
                        text=text,
                        line=dict(color=colors[color_lookup], dash=dash)
                    )
                    plot_data.append(trace)

            if len(plot_data) > 0:
                layout = go.Layout(
                    title=f'Model and Annotation plots {story_id}',
                    hovermode='closest',
                    xaxis=dict(
                        # title='Position',
                    ),
                    yaxis=dict(
                        title=f"Suspense",
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h")
                )

                fig = go.Figure(data=plot_data, layout=layout)

                export_plots(args, f"/model_annotation_plots/{story_id}", fig)

def evaluate_stories(args):
    print(f"Evaluate stories: {args}")

    ensure_dir(f"{args['output_dir']}/sentence_model_evaluation/")

    position_df = pd.read_csv(args["position_stats"])
    print(f"Position rows : {len(position_df)}")
    annotator_df = pd.read_csv(args["annotator_targets"])

    if args["exclude_worker_ids"] is not None and len(args["exclude_worker_ids"]) > 0:
        annotator_df = annotator_df[~annotator_df["worker_id"].isin(args["exclude_worker_ids"])]

    print(f"Annotator rows: {len(annotator_df)}")

    vector_df = None
    if "vector_stats" in args and len(args["vector_stats"]) > 0:
        vector_df = pd.read_csv(args["vector_stats"])

    plot_annotator_and_model_predictions(position_df, annotator_df, args)
    contineous_evaluation(position_df, annotator_df, args)
    ordinal_regression_bucketed_evaluation(position_df, annotator_df, args)

def scale_prediction_columns(position_df):
    for col in model_prediction_columns:
        for feature_col in [f"{col}_diff", f"{col}"]:

            if feature_col not in position_df.columns:
                continue

            scaler = StandardScaler()
            scaled_col = numpy.squeeze(scaler.fit_transform(position_df[feature_col].to_numpy().reshape(-1, 1)), axis=1).tolist()
            position_df[f"{feature_col}_scaled"] = scaled_col
    return position_df


def prepare_dataset(annotator_df, position_df, keep_first_sentence=False):
    merged_df = pd.merge(position_df, annotator_df, left_on=["story_id", "sentence_num"],
                         right_on=["story_id", "sentence_num"], how="inner")
    merged_df = merged_df.sort_values(by=["worker_id", "story_id", "sentence_num"]).reset_index()
    merged_df = pd.concat([merged_df, merged_df[1:].reset_index(drop=True).add_suffix("_later")],
                          axis=1)
    merged_df = merged_df.sort_values(by=["worker_id", "story_id", "sentence_num"]).reset_index()
    merged_df = merged_df.loc[merged_df["story_id"] == merged_df["story_id_later"]]
    merged_df = merged_df.loc[merged_df["worker_id"] == merged_df["worker_id_later"]]
    merged_df = merged_df.loc[merged_df["sentence_num"] + 1 == merged_df["sentence_num_later"]]
    for col in model_prediction_columns:
        merged_df[f"{col}_diff"] = merged_df[f"{col}_later"] - merged_df[col]

    merged_df = scale_prediction_columns(merged_df)
    print(f"Merged rows: {len(merged_df)}")
    # Remove the first line as cannot be judged relatively.

        # if not keep_first_sentence:
    if keep_first_sentence:
        merged_df = merged_df.loc[(merged_df["suspense"] != 0.0) | (merged_df["sentence_num"] == 0)]
    else:
        merged_df = merged_df.loc[merged_df["suspense"] != 0.0]

    max_df = merged_df.groupby(by=["story_id"], as_index=False)["sentence_num"].max()
    df_all = merged_df.merge(max_df, on=['story_id', 'sentence_num'],
                             how='left', indicator=True)
    merged_df = df_all.loc[df_all['_merge'] == "left_only"]

    merged_df = merged_df.sort_values(by=["story_id","worker_id","sentence_num"]).reset_index()

    return merged_df


def export_plots(args, file, fig):
    ensure_dir(f"{args['output_dir']}/{file}")
    if not args["no_html_plots"]:
        file_path = f"{args['output_dir']}/{file}.html"
        print(f"Save plot: {file_path}")
        pio.write_html(fig, file_path)
    if not args["no_pdf_plots"]:
        file_path = f"{args['output_dir']}/{file}.pdf"
        print(f"Save plot pdf: {file_path}")
        pio.write_image(fig, file_path)

evaluate_stories(vars(args))
