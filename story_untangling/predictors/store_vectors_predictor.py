from collections import OrderedDict

import pandas
import torch
from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from anytree import AnyNode, Node, PreOrderIter
from anytree.exporter import DictExporter

from story_untangling.predictors.uncertain_reader_gen_predictor import only_tensor_nodes

full_stop_token = 239
exporter = DictExporter(dictcls=OrderedDict, attriter=sorted)


@Predictor.register("uncertain_reader_store_vector_predictor")
class UncertainReaderStoreVectorPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)

        story_id_file = "/afs/inf.ed.ac.uk/group/project/comics/stories/WritingPrompts/annotation_results/raw/story_id_dev_splitad"

        story_id_df = pandas.read_csv(story_id_file)
        self.story_ids_to_predict = set(story_id_df['story_id'])
        self.only_annotation_stories = False

        self._remove_sentence_output = False

        self._model.full_output_embedding = True
        self._model.run_feedforwards = True  # Turn off normal feedforwards to avoid running twice.

        self.embedder = model._text_field_embedder._token_embedders["openai_transformer"]
        self.embedder._top_layer_only = True

        self.dataset_reader = dataset_reader
        self.dataset_reader.story_chunking = 150  # Allow bigger batching for sampling.
        self.tokenizer = dataset_reader._tokenizer
        self.indexer = dataset_reader._token_indexers["openai_transformer"]


        self.dataset_reader = dataset_reader
        self.dataset_reader._story_chunking = 200  # Allow bigger batching for sampling.
        self.dataset_reader._marked_sentences = True

        self._device = self._model._lm_model._decoder.weight.device

    def predict_instance(self, instance: Instance) -> JsonDict:

        story_id = instance["metadata"]["story_id"]
        if self.only_annotation_stories and story_id not in self.story_ids_to_predict:
            print(f"Skipping non annotation story: {story_id}")
            return None

        print(f"Predicting story: {story_id}")

        outputs = self._model.forward_on_instance(instance)

        root = self.sample_tree(instance, outputs)

        for n in PreOrderIter(root, only_tensor_nodes):

            for attr, value in n.__dict__.items():
                print(attr, value)
                if isinstance(value, torch.Tensor):
                    if len(value.shape) == 0:
                        n.__dict__[attr] = value.item()
                    else:
                        n.__dict__[attr] = value.tolist()

        root_as_dict = exporter.export(root)
        return sanitize(root_as_dict)

    def sample_tree(self, instance, outputs):
        ''' Create a hierarchy of possibilities by generating sentences in a tree structure.
        '''
        embedded_text_tensor = outputs["embedded_text_tensor"]
        embedded_text_tensor = torch.from_numpy(embedded_text_tensor)
        embedded_text_mask = torch.from_numpy(outputs["masks"]).long()
        encoded_stories = torch.from_numpy(outputs["source_encoded_stories"])

        encoded_sentences = torch.from_numpy(outputs["sentence_tensor"])

        text = instance["text"]

        root = Node(name="representations")

        for position, (text_field, encoded_story) in enumerate(zip(text, encoded_stories)):

            if position == embedded_text_tensor.shape[0]:
                break

            sentence_length = int(torch.sum(embedded_text_mask[position]).item())

            if sentence_length == 0:
                continue

            story_id = instance["metadata"]["story_id"]
            sentence_ids = instance["metadata"]["sentence_ids"]
            sentence_nums = instance["metadata"]["sentence_nums"]
            text = instance["metadata"]["text"]

            if position >= len(sentence_ids):
                continue

            sentence_id = sentence_ids[position]
            sentence_num = sentence_nums[position]
            sentence_text = text[position]

            encoded_story_tensor = encoded_stories[position].detach().cpu()

            encoded_sentence = torch.squeeze(encoded_sentences, dim=0)[position].detach().cpu()

            position_node = AnyNode(position=f"{position}", story_id=story_id,
                                    sentence_id=sentence_id,
                                    sentence_num=sentence_num,
                                    story_tensor=encoded_story_tensor,
                                    sentence_tensor=encoded_sentence,
                                    sentence_text=sentence_text,
                                    sentence_length=sentence_length,
                                    parent=root)

        return root
