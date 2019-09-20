from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from story_untangling.predictors.local_beam_pairwise_ordering_predictor import ReadingThoughtsLocalGreedyPredictor


@Predictor.register("global_beam_pairwise_ordering_predictor")
class ReadingThoughtsGlobalBeamPredictor(ReadingThoughtsLocalGreedyPredictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader, language)
        self._model._full_output_score = True
        self._exclude_first = False
