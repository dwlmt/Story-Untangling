import copy
from math import log

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

from story_untangling.predictors.local_greedy_pairwise_ordering_predictor import ReadingThoughtsLocalGreedyPredictor
from story_untangling.predictors.welford import Welford


@Predictor.register("global_beam_pairwise_ordering_predictor")
class ReadingThoughtsGlobalBeamPredictor(ReadingThoughtsLocalGreedyPredictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm', beam_size=10000) -> None:
        super().__init__(model, dataset_reader, language)
        self._beam_size = beam_size
        self._model._full_output_score = True
        self._exclude_first = False


    def search(self, predicted_sentence_lookup, shuffled_instances):

        instance_results = self._model.forward_on_instances(shuffled_instances)

        all_probs = [p["neighbour_log_probs"].tolist() for p in instance_results]

        print(all_probs)

        # Put all initial starting positions into the list
        hypotheses = [([r], 0.0) for r in range(len(shuffled_instances))]


        # Go to the required length.
        for i in range(len(shuffled_instances) - 1):
            fringe_sequences = []
            for seq, score in hypotheses:
                for j, prob in [(i, p) for i, p in enumerate(all_probs[seq[-1]]) if i not in set(seq)]:
                    fringe_candidate = (seq + [j], score + prob)

                    fringe_sequences.append(fringe_candidate)
            ordered = sorted(fringe_sequences, key=lambda tup: tup[1], reverse=True)
            hypotheses = ordered[:self._beam_size]

        best_sequence, _ = hypotheses[0]

        predicted_order = [shuffled_instances[s]["metadata"]["absolute_position"] for s in best_sequence]

        return predicted_order


