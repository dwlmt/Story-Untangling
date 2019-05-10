from collections import OrderedDict

import torch
from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from anytree import AnyNode, Node, PreOrderIter
from anytree.exporter import DictExporter

full_stop_token = 239
exporter = DictExporter(dictcls=OrderedDict, attriter=sorted)


def random_sample(logits: torch.Tensor) -> int:
    d = torch.distributions.Categorical(logits=logits)
    sampled = d.sample()

    return sampled.item()


def choose_max(logits: torch.Tensor, ) -> int:
    p, i = torch.max(logits, dim=-1)
    return i.item()



@Predictor.register("uncertain_reader_gen_predictor")
class UncertainReaderGenPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)

        self._model.full_output_embedding = True

        self.embedder = model._text_field_embedder._token_embedders["openai_transformer"]
        self.embedder._top_layer_only = True

        self.tokenizer = dataset_reader._tokenizer
        self.indexer =  dataset_reader._token_indexers["openai_transformer"]

        self.sentences_to_rollout = 1
        self.samples_per_level = 3
        self.keep_tensor_output = False


    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        embedded_text_tensor = outputs["embedded_text_tensor"]
        embedded_text_tensor = torch.from_numpy(embedded_text_tensor).cpu()
        embedded_text_mask = torch.from_numpy(outputs["masks"]).cpu()

        encoded_stories = torch.from_numpy(outputs["source_encoded_stories"]).cpu()
        text = instance["text"]

        root = Node(name="root")
        per_sentence = Node(name="sentences", parent=root)
        aggregate = Node(name="aggregate_stats", parent=root)

        for position, (text_field, encoded_story) in enumerate(zip(text, encoded_stories)):


            text_tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
            text_to_gen_from = text_tensor_dict['openai_transformer']
            ctx_size = text_to_gen_from.size(0)
            sentence_length = torch.sum(embedded_text_mask[position]).item()
            text_to_gen_from = text_to_gen_from[0:sentence_length]

            if sentence_length == 0:
                continue

            position_node = AnyNode(name=f"{position}", story_id=instance["metadata"]["story_id"],
                                    sentence_id=instance["metadata"]["sentence_ids"][position],
                                    sentence_num=instance["metadata"]["sentence_nums"][position],
                                    parent=per_sentence)
            # Use a paths placeholder so the aggregate statsitics can be separated from the
            paths = Node(name="paths", parent=position_node)
            stats = Node(name="stats", parent=position_node)

            # This is the correct answer from the story.
            correct_base = AnyNode(gold=True, story_tensor=encoded_story.clone().cpu().detach(),
                                   sentence_ids=text_to_gen_from.tolist(),
                                   sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                  text_to_gen_from.tolist() if t in self.indexer.decoder],
                                   sentence_length=sentence_length, parent=paths)

            # Put the correct representations into the sentences at the given point.
            for i in range(1, self.sentences_to_rollout + 1):

                if position + i > len(text) - 1 or len(text[position + i]) == 0:
                    continue

                text_tensor_future_dict = text[position + i].as_tensor(text_field.get_padding_lengths())
                text_to_gen_from_future = text_tensor_future_dict['openai_transformer']

                sentence_length_future = torch.sum(embedded_text_mask[position + i]).item()
                text_to_gen_from_future = text_to_gen_from_future[0:sentence_length_future]

                correct_base = AnyNode(gold=True,
                                       story_tensor=encoded_stories[position + i].cpu().detach(),
                                       sentence_ids=text_to_gen_from_future.tolist(),
                                       sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                      text_to_gen_from_future.tolist() if
                                                      t in self.indexer.decoder],
                                       sentence_length=sentence_length_future,
                                       parent=correct_base)
                # print(f"Add Node: {correct_base}")


            for sample_num in range(self.samples_per_level):
                self.generate_sentence(position, embedded_text_tensor, embedded_text_mask, encoded_story,
                                       text_to_gen_from,
                                       sentence_length, ctx_size, recursion=self.sentences_to_rollout - 1, parent=paths)

        for n in PreOrderIter(root):
            if isinstance(n, AnyNode) and "story_tensor" in n.__dict__:
                if self.keep_tensor_output:
                    n.__dict__["story_tensor"] = n.__dict__["story_tensor"].tolist()
                else:
                    n.__dict__.pop("story_tensor", None)

        #print(RenderTree(root))
        root_as_dict = exporter.export(root)
        return sanitize(root_as_dict)

    def generate_sentence(self, position, embedded_text_tensor, embedded_text_mask, encoded_story,
                          text_to_gen_from, sentence_length, ctx_size, recursion=0, parent=None):

        device = self._model._lm_model._decoder.weight.device

        gen_sentence = []

        encoded_text_merged = embedded_text_tensor[position, :, :]

        for i in range(ctx_size - 1):

            if len(gen_sentence) > 0:

                text_tensor_merged = torch.cat((text_to_gen_from, torch.tensor(gen_sentence).long()))

                sentence_length = text_tensor_merged.shape[-1]

                if text_tensor_merged.shape[0] > ctx_size:
                    text_tensor_merged = (text_tensor_merged[1:ctx_size + 1])
                else :
                    text_tensor_merged = torch.cat(
                        (text_tensor_merged, torch.tensor([0] * (ctx_size - len(text_tensor_merged))).long()))

                encoded_text_merged = self.embedder(torch.unsqueeze(text_tensor_merged.to(device), dim=0).long())

            next_word_id = self.predict(encoded_text_merged, encoded_story, sentence_length, ctx_size)

            if not next_word_id:
                break

            gen_sentence.append(next_word_id)

            if next_word_id == full_stop_token:
                break

        # Pad up to the context length.
        gen_sentence_length = len(gen_sentence)
        if gen_sentence_length < ctx_size:
            gen_sentence_padded = gen_sentence + ([0] * (ctx_size - len(gen_sentence)))
        else:
            gen_sentence_padded = gen_sentence

        gen_sentence_padded = torch.tensor(gen_sentence_padded).long()
        gen_sentence_padded = torch.unsqueeze(gen_sentence_padded, dim=0)

        generated_sentence_tensor = self.embedder(gen_sentence_padded.to(device))

        new_embedded_text_tensor = torch.zeros(embedded_text_tensor.shape[0], max(embedded_text_tensor.shape[1],
                                                                                  generated_sentence_tensor.shape[1]),
                                               embedded_text_tensor.shape[2])
        new_embedded_text_tensor[:, 0 : embedded_text_tensor.shape[1], :] = embedded_text_tensor
        new_embedded_text_tensor[position:, 0: generated_sentence_tensor.shape[1], :] = generated_sentence_tensor
        embedded_text_tensor = new_embedded_text_tensor

        new_embedded_text_mask = torch.zeros(embedded_text_tensor.shape[0], embedded_text_tensor.shape[1])
        new_embedded_text_mask[: , 0: embedded_text_mask.shape[1]] = embedded_text_mask
        embedded_text_mask = new_embedded_text_mask.long()

        encoded_sentences, encoded_story, story_sentence_mask = self._model.encode_story_vectors(
            embedded_text_tensor.to(device), embedded_text_mask.to(device))

        encoded_story = encoded_story[position]

        # TODO: Need to integrate with prob generation.
        # if self._model._story_feedforward:
        #    encoded_story = self._model._story_feedforward(encoded_story)

        encoded_story = torch.squeeze(encoded_story, dim=0)

        encoded_story = torch.squeeze(encoded_story)

        created_node = AnyNode(gold=False, story_tensor=encoded_story.cpu().detach(), sentence_ids=gen_sentence,
                               sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                              gen_sentence if t in self.indexer.decoder],
                               sentence_length=len(gen_sentence),
                               parent=parent)

        #print(created_node)

        if recursion > 0:
            for i in range(self.samples_per_level):

                next_position = position + 1

                # Don't project past the end of the story.
                if next_position == embedded_text_tensor.shape[0]:
                    break

                self.generate_sentence(next_position, embedded_text_tensor, embedded_text_mask, encoded_story,
                                       torch.tensor(gen_sentence).long(), len(gen_sentence), ctx_size, recursion=recursion-1)

    def predict(self, embedded_text, story, sentence_length, ctx_size):
        device = self._model._lm_model._decoder.weight.device
        story = torch.unsqueeze(story, dim=0)
        logits = self._model._lm_model(embedded_text.to(device), story.to(device))
        logits =  torch.squeeze(logits, dim=0)
        logits = logits[min(sentence_length - 1, ctx_size - 1), :]
        next_word_id = random_sample(logits)

        return next_word_id

