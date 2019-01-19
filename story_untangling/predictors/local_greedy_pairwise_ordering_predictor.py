import copy
from collections import OrderedDict

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


@Predictor.register("local_greedy_pairwise_ordering_predictor")
class ReadingThoughtsLocalGreedyPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)
        self._model._full_output_score = True

        self._spearmanr_wel = Welford()
        self._kendalls_tau_wel = Welford()

        self._pmr_correct = 0.0
        self._pmr_total = 0.0

        self._pos_acc_correct = 0.0
        self._pos_acc_total = 0.0

        self._spearmanr_p_values = []
        self._kendalls_tau_p_values = []

        self._exclude_first = True

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:

        gold_instances = instances

        shuffled_instances = instances.copy()

        if self._exclude_first:
            shuffled_tail = shuffled_instances[2:]
            random.shuffle(shuffled_tail)
            shuffled_instances[2:] = shuffled_tail
        else:
            random.shuffle(shuffled_instances)


        story_ids = []
        gold_order = []
        for instance in gold_instances:
            story_ids.append(instance["metadata"]["story_id"])
            gold_order.append(instance["metadata"]["absolute_position"])

        # Do not split the ordering task across stories.
        if len(set(story_ids)) > 1:
            return {}

        shuffled_sentence_order = []
        predicted_sentence_lookup = {}
        for i, instance in enumerate(shuffled_instances):
            shuffled_sentence_order.append(instance["metadata"]["absolute_position"])
            predicted_sentence_lookup[instance["metadata"]["absolute_position"]] = i

        results = {}

        predicted_order = self.search(gold_order,  predicted_sentence_lookup, shuffled_instances)

        if self._exclude_first:
            gold_order_to_eval = gold_order[1:]
            predicted_order_to_eval = predicted_order[1:]
        else:
            gold_order_to_eval = gold_order
            predicted_order_to_eval = predicted_order


        kendalls_tau, kendalls_tau_p_value = stats.kendalltau(gold_order_to_eval, predicted_order_to_eval)
        spearmanr, spearmanr_p_value = stats.spearmanr(gold_order_to_eval, predicted_order_to_eval)

        self._spearmanr_wel(spearmanr)
        self._kendalls_tau_wel(kendalls_tau)

        self._spearmanr_p_values.append(spearmanr_p_value)
        self._kendalls_tau_p_values.append(kendalls_tau_p_value)

        if gold_order_to_eval == predicted_order_to_eval:
            self._pmr_correct += 1.0
        self._pmr_total += 1.0

        self._pos_acc_correct += [a == b for a, b in
                                  zip(gold_order_to_eval, predicted_order_to_eval)].count(True)
        self._pos_acc_total += len(gold_order_to_eval)


        results["initial_ordering"] = shuffled_sentence_order
        results["gold_ordering"] = gold_order
        results["predicted_ordering"] = predicted_order
        results["text_reference"] = OrderedDict({i["metadata"]["absolute_position"] : i["metadata"]["source_text"] for i in gold_instances})

        results["kendalls_tau"] = kendalls_tau
        results["kendalls_tau_p_value"] = kendalls_tau_p_value
        results["spearmanr"] = spearmanr
        results["spearmanr_p_value"] = spearmanr_p_value

        results["kendalls_tau_culm_avg"], results["kendalls_tau_culm_std"] = self._kendalls_tau_wel.meanfull
        results["spearmanr_culm_avg"], results["spearmanr_culm_std"] = self._spearmanr_wel.meanfull
        results["perfect_match_ratio_culm"] = self._pmr_correct / self._pmr_total
        results["position_accuracy_culm"] = self._pos_acc_correct / self._pos_acc_total

        return [results]

    def search(self, gold_order, predicted_sentence_lookup,  shuffled_instances):

        predicted_order = []
        predicted_order.append(shuffled_instances[0]["metadata"]["absolute_position"])

        instance_results = self._model.forward_on_instances(shuffled_instances)
        while len(predicted_order) < len(gold_order):
            last_predicted_index = predicted_order[-1]

            neighbour_scores = instance_results[predicted_sentence_lookup[last_predicted_index]]["neighbour_scores"]
            # Don't link a sentence with itself.
            predicted_sentence_indices = [predicted_sentence_lookup[i] for i in predicted_order]
            max_score = max([ns for i, ns in enumerate(neighbour_scores) if i not in predicted_sentence_indices])
            max_index = numpy.where(neighbour_scores == max_score)
            predicted_order.append(shuffled_instances[numpy.asscalar(max_index[0])]["metadata"]["absolute_position"])

        return predicted_order
