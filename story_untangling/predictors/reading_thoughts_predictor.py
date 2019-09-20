from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("reading_thoughts_predictor")
class ReadingThoughtsPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._model._full_output_embeddings = False

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        print(json_dict)
