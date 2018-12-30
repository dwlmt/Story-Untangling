from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.common.util import get_spacy_model
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from story_untangling.dataset_readers.dataset_utils import dual_window


@Predictor.register("reading_thoughts_predictor")
class ReadingThoughtsPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        # We have to use spacy to tokenise our document here, because we need
        # to also know sentence boundaries to propose valid mentions.
        self._spacy = get_spacy_model(language, pos_tags=True, parse=True, ner=False)
        self._story_id = 0

    def predict(self, story: str, context_sentences: int = 2, prediction_sentences: int = 1) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        Parameters
        ----------
        story : ``str``
            A string representation of a story.
        context_sentences : ``int``
            How many context sentences in the sliding window.
        prediction_sentences : ``int``
            How many sentences to predict in the sliding window.
        Returns
        -------
        A dictionary representation with sliding window predictions with similarities.
        """
        return self.predict_json({"story" : story, "context_sentences" : context_sentences, "prediction_sentences" : prediction_sentences})

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
        instances = self._json_to_instance(inputs)
        results_list = []
        results = {"results" : results_list}

        for instance in instances:
            results_list.append(self.predict_instance(instance))

        return results


    def _json_to_instance(self, json_dict: JsonDict) -> List[Instance]:
        """
        Expects JSON that looks like ``{"story": "string of the story text"}``
        """
        instances = []
        story = json_dict["story"]
        context_sentences = json_dict.get("context_sentences", 2)
        prediction_sentences = json_dict.get("prediction_sentences", 1)
        spacy_document = self._spacy(story)
        sentences = [{"sentence_num": i, "text": sentence.text} for i, sentence in enumerate(spacy_document.sents)]

        sentence_nums = [i for i in range(len(sentences))]

        for source_indices, target_indices, absolute_position, relative_position in dual_window(sentence_nums,
                                                                                                context_size=context_sentences,
                                                                                                predictive_size=prediction_sentences,
                                                                                                num_of_sentences=len(sentences)):

            source_sequence = [sentences[i] for i in source_indices if i is not None]
            target_sequence = [sentences[i] for i in target_indices if i is not None]
            negative_sequence = None

            metadata = {"story_id": self._story_id, "absolute_position": absolute_position,
                        "relative_position": relative_position, "num_of_sentences": len(sentences)}

            instance = self._dataset_reader.text_to_instance(source_sequence, target_sequence, negative_sequence,
                                        metadata=metadata)
            instances.append(instance)

        self._story_id += 1

        return instances