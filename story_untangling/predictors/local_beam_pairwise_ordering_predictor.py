import copy
from collections import OrderedDict
from math import exp
from time import sleep

import numpy
import random
from typing import List, Any, Dict

from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import sanitize
from scipy.stats import stats

from story_untangling.predictors.welford import Welford


@Predictor.register("local_beam_pairwise_ordering_predictor")
class ReadingThoughtsLocalGreedyPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)
        self._model._full_output_score = True
        self._beam_size = 1000
        self._best_n = 5

        self._spearmanr_wel = Welford()
        self._kendalls_tau_wel = Welford()
        self._pearsonr_wel = Welford()

        self._pmr_correct = 0.0
        self._pmr_total = 0.0

        self._pos_acc_correct = 0.0
        self._pos_acc_total = 0.0

        self._spearmanr_p_values = []
        self._kendalls_tau_p_values = []
        self._pearsonr_p_values = []

        self._exclude_first = True

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:

        gold_instances = copy.copy(instances)
        shuffled_instances = copy.copy(instances)

        if self._exclude_first:
            shuffled_tail = shuffled_instances[2:]
            random.shuffle(shuffled_tail)
            shuffled_instances[2:] = shuffled_tail
        else:
            random.shuffle(shuffled_instances)


        story_ids = []
        gold_order = []
        for i, instance in enumerate(gold_instances):
            story_ids.append(instance["metadata"]["story_id"])
            gold_order.append(instance["metadata"]["absolute_position"])


        # Do not split the ordering task across stories.
        if len(set(story_ids)) > 1:
            return {}

        shuffled_sentence_order = []
        shuffled_sentence_pos_to_idx = {}
        for i, instance in enumerate(shuffled_instances):
            shuffled_sentence_order.append(instance["metadata"]["absolute_position"])
            shuffled_sentence_pos_to_idx[instance["metadata"]["absolute_position"]] = i
        shuffled_sentence_idx_to_pos = {v: k for k, v in shuffled_sentence_pos_to_idx.items()}


        results = {}

        predicted_order, log_prob, best_n = self.search(shuffled_instances, shuffled_sentence_pos_to_idx, shuffled_sentence_idx_to_pos)

        if predicted_order is None:
            return {}

        if self._exclude_first:
            gold_order_to_eval = gold_order[1:]
            predicted_order_to_eval = predicted_order[1:]
        else:
            gold_order_to_eval = gold_order
            predicted_order_to_eval = predicted_order


        kendalls_tau, kendalls_tau_p_value = stats.kendalltau(gold_order_to_eval, predicted_order_to_eval)
        spearmanr, spearmanr_p_value = stats.spearmanr(gold_order_to_eval, predicted_order_to_eval)
        pearsonr, pearsonr_p_value = stats.pearsonr(gold_order_to_eval, predicted_order_to_eval)

        self._spearmanr_wel(spearmanr)
        self._kendalls_tau_wel(kendalls_tau)
        self._pearsonr_wel(pearsonr)

        self._spearmanr_p_values.append(spearmanr_p_value)
        self._kendalls_tau_p_values.append(kendalls_tau_p_value)
        self._pearsonr_p_values.append(pearsonr_p_value)

        if gold_order_to_eval == predicted_order_to_eval:
            self._pmr_correct += 1.0
        self._pmr_total += 1.0

        self._pos_acc_correct += [a == b for a, b in
                                  zip(gold_order_to_eval, predicted_order_to_eval)].count(True)
        self._pos_acc_total += len(gold_order_to_eval)


        results["initial_ordering"] = shuffled_sentence_order
        results["gold_ordering"] = gold_order
        results["predicted_ordering"] = predicted_order
        results["best_answer_log_prob"] = log_prob
        results["best_answer_prob"] = exp(log_prob)
        results["best_n"] = best_n

        results["kendalls_tau"] = kendalls_tau
        results["kendalls_tau_p_value"] = kendalls_tau_p_value
        results["spearmanr"] = spearmanr
        results["spearmanr_p_value"] = spearmanr_p_value
        results["pearsonr"] = pearsonr
        results["pearsonr_p_value"] = pearsonr_p_value

        results["kendalls_tau_culm_avg"], results["kendalls_tau_culm_std"] = self._kendalls_tau_wel.meanfull
        results["spearmanr_culm_avg"], results["spearmanr_culm_std"] = self._spearmanr_wel.meanfull
        results["pearsonr_culm_avg"], results["pearsonr_culm_std"] = self._pearsonr_wel.meanfull
        results["perfect_match_ratio_culm"] = self._pmr_correct / self._pmr_total
        results["position_accuracy_culm"] = self._pos_acc_correct / self._pos_acc_total

        results["source_text"] = OrderedDict(
            {i["metadata"]["absolute_position"]: i["metadata"]["source_text"] for i in gold_instances})
        results["target_text"] = OrderedDict(
            {i["metadata"]["absolute_position"]: i["metadata"]["target_text"] for i in gold_instances})

        return [results]

    def search(self, shuffled_instances, shuffled_sentence_pos_to_idx, shuffled_sentence_idx_to_pos):

        #TODO: This wouldn't handle a sliding window with a step of more than one so this would need to be changed.

        max_pos = max(shuffled_sentence_pos_to_idx, key=int)

        instance_results = self._model.forward_on_instances(shuffled_instances)
        all_probs = [p["neighbour_log_probs"].tolist() for p in instance_results]
        # Put all initial starting positions into the list
        if self._exclude_first:
            hypotheses = [([shuffled_sentence_idx_to_pos[0]], 0.0)]
        else:
            hypotheses = [([shuffled_sentence_idx_to_pos[r]], 0.0) for r in range(len(shuffled_instances))]

        # Go to the required length.
        for i in range(len(shuffled_instances) - 1):
            fringe_sequences = []
            for seq, score in hypotheses:
                for j, log_prob in [(i, p) for i, p in enumerate(all_probs[shuffled_sentence_pos_to_idx[seq[-1]]])]:
                    next_pos = min(max_pos, shuffled_sentence_idx_to_pos[j] + 1)
                    if next_pos not in set(seq):

                        # Plus one is needed because the correct target needs to redirect to the next in the sequence.
                        fringe_candidate = (seq + [next_pos], score + log_prob)
                        fringe_sequences.append(fringe_candidate)
            ordered = sorted(fringe_sequences, key=lambda tup: tup[1], reverse=True)
            hypotheses = ordered[:self._beam_size]

        if len(hypotheses) == 0:
            return None, None, None

        best_n = []
        for seq, log_prob in hypotheses[0:self._best_n]:
            best_n.append({"predicted_order": seq, "log_prob": log_prob, "prob": exp(log_prob)})

        best_sequence, log_prob = hypotheses[0]

        return best_sequence, log_prob, best_n
