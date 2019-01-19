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
        self._exclude_first = True


    def search(self, gold_order, predicted_sentence_lookup, shuffled_instances):

        instance_results = self._model.forward_on_instances(shuffled_instances)

        probs = [p["neighbour_probs"].tolist() for p in instance_results]

        sequences = [[list(), 1.0]]

        for row in probs:
            all_candidates = list()

            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    if j not in set(seq):
                        candidate = [seq + [j], score * -log(row[j])]
                        all_candidates.append(candidate)


            ordered = sorted(all_candidates, key=lambda tup: tup[1])

            sequences = ordered[:self._beam_size]

        best_sequence = sequences[0][0]

        predicted_order = [shuffled_instances[s]["metadata"]["absolute_position"] for s in best_sequence]

        return predicted_order


