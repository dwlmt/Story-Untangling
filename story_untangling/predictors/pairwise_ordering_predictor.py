from typing import List, Any, Dict


from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from scipy.stats import stats

from story_untangling.predictors.welford import Welford


@Predictor.register("pairwise_ordering_predictor")
class ReadingThoughtsPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)

        self._spearmanr_wel = Welford()
        self._kendalls_tau_wel = Welford()

        self._pmr_correct = 0.0
        self._pmr_total = 0.0

        self._pos_acc_correct = 0.0
        self._pos_acc_total = 0.0

        self._spearmanr_p_values = []
        self._kendalls_tau_p_values = []



    def predict(self, story: Dict[str, Any]) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        Parameters
        ----------
        story : ``dict``
            A string representation of a story.
        Returns
        -------
        A dictionary representation with sliding window predictions with similarities.
        """
        return self.predict_json({"story": story})

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        """
        Override the default batch instances as generates a sliding window over a story.
        So a single input text
        """
        instances = []
        for json_dict in json_dicts:
            instances.extend(self._json_to_instance(json_dict))
        return instances

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        sentences = inputs["sentences"]
        initial_sentence_order = [s["sentence_num"] for s in sentences]
        initial_sentence_order_text = [s["text"] for s in sentences]
        gold_index = sorted(((v, i) for i, v in enumerate(initial_sentence_order)))
        gold_order = [initial_sentence_order_text[i] for i, i in gold_index]

        instances = self._json_to_instance(inputs)

        results = {}

        instance_results = self.predict_batch_instance(instances)

        predicted_sentence_indices = []
        # The numbers are the indices positions not the actual sentence num in the story.
        predicted_sentence_indices.append(0)  # Add the first one as is always correct as unshuffled.

        while len(predicted_sentence_indices) < len(gold_index):
            last_predicted_index = predicted_sentence_indices[-1]
            neighbour_scores = instance_results[last_predicted_index]["neighbour_scores"]
            # Don't link a sentence with itself.
            max_index = neighbour_scores.index(
                max([ns for i, ns in enumerate(neighbour_scores) if i not in predicted_sentence_indices]))
            predicted_sentence_indices.append(max_index)

        # Based on the indices select the sentence number.
        predicted_sentence_order = [initial_sentence_order[i] for i in predicted_sentence_indices]

        gold_order_not_first = gold_order[1:]
        predicted_sentence_order_not_first = predicted_sentence_order[1:]

        kendalls_tau, kendalls_tau_p_value = stats.kendalltau(gold_order_not_first, predicted_sentence_order_not_first)
        spearmanr, spearmanr_p_value = stats.spearmanr(gold_order_not_first, predicted_sentence_order_not_first)

        self._spearmanr_wel(spearmanr)
        self._kendalls_tau_wel(kendalls_tau)

        self._spearmanr_p_values.append(spearmanr_p_value)
        self._kendalls_tau_p_values.append(kendalls_tau_p_value)

        if gold_order_not_first == predicted_sentence_order_not_first:
            self._pmr_correct += 1.0
        self._pmr_total += 1.0

        self._pos_acc_correct += [a == b for a, b in
                                  zip(gold_order_not_first, predicted_sentence_order_not_first)].count(True)
        self._pos_acc_total += len(gold_order_not_first)


        results["initial_ordering"] = initial_sentence_order
        results["gold_ordering"] = [g for g, _ in gold_index]
        results["predicted_ordering"] = predicted_sentence_order
        results["gold_text"] = gold_order
        results["predicted_text"] = [initial_sentence_order_text[i] for i in predicted_sentence_indices]

        results["kendalls_tau"] = kendalls_tau
        results["kendalls_tau_p_value"] = kendalls_tau_p_value
        results["spearmanr"] = spearmanr
        results["spearmanr_p_value"] = spearmanr_p_value

        results["kendalls_tau_culm_avg"], results["kendalls_tau_culm_std"] = self._kendalls_tau_wel.meanfull
        results["spearmanr_culm_avg"], results["spearmanr_culm_std"] = self._spearmanr_wel.meanfull
        results["perfect_match_ratio_culm"] = self._pmr_correct / self._pmr_total
        results["position_accuracy_culm"] = self._pos_acc_correct / self._pos_acc_total

        return results

    def _json_to_instance(self, json_dict: JsonDict) -> List[Instance]:
        """
        Expects JSON that looks like ``{"story": "string of the story text"}``
        """
        instances = list()
        story_id = json_dict["id"]

        sentences = json_dict["sentences"]

        for sentence in sentences:
            metadata = {"story_id": story_id, "absolute_position": sentence["sentence_num"],
                        "relative_position": sentence["sentence_num"] / float(json_dict["sentence_num"]),
                        "num_of_sentences": json_dict["sentence_num"],
                        "full_output_score": True}

            instance = self._dataset_reader.text_to_instance([sentence], [sentence], metadata=metadata)
            instances.append(instance)

        return instances
