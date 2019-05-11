import copy
from collections import OrderedDict

import torch
from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from anytree import AnyNode, Node, PreOrderIter
from anytree.exporter import DictExporter
from torch.nn import Softmax, PairwiseDistance

full_stop_token = 239
exporter = DictExporter(dictcls=OrderedDict, attriter=sorted)


def random_sample(logits: torch.Tensor) -> int:
    d = torch.distributions.Categorical(logits=logits)
    sampled = d.sample()

    return sampled.item()


def choose_max(logits: torch.Tensor, ) -> int:
    p, i = torch.max(logits, dim=-1)
    return i.item()


def only_tensor_nodes(node):
    if isinstance(node, AnyNode) and hasattr(node, "story_tensor"):
        return True
    else:
        return False


@Predictor.register("uncertain_reader_gen_predictor")
class UncertainReaderGenPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)

        self._model.full_output_embedding = True
        self._model.run_feedforwards = False  # Turn off normal feedforwards to avoid running twice.

        self.embedder = model._text_field_embedder._token_embedders["openai_transformer"]
        self.embedder._top_layer_only = True

        self.tokenizer = dataset_reader._tokenizer
        self.indexer =  dataset_reader._token_indexers["openai_transformer"]

        self.sentences_to_rollout = 3
        self.samples_per_level = 3
        self.keep_tensor_output = False

        self._device = self._model._lm_model._decoder.weight.device

        self._softmax = Softmax(dim=-1)

        self._l1 = PairwiseDistance(p=1.0)
        self._l2 = PairwiseDistance(p=2.0)


    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        root = self._sample_tree(instance, outputs)

        for n in PreOrderIter(root, only_tensor_nodes):

            for attr, value in n.__dict__.items():
                if isinstance(value, torch.Tensor):
                    n.__dict__[attr] = value.tolist()

            if not self.keep_tensor_output:
                n.story_tensor = None

        root_as_dict = exporter.export(root)
        return sanitize(root_as_dict)

    def _calculate_disc_probabilities(self, root):
        ''' Iterate over the tree and calculate the probabilities for each path using the discriminate loss function.
        '''
        gold_index = 0
        child_list = []

        for child in PreOrderIter(root, only_tensor_nodes, maxlevel=2):
            if child.parent == root:
                child_list.append(child)

        if len(child_list) == 0:
            return

        parent_story_tensor = root.story_tensor
        parent_story_tensor = torch.unsqueeze(parent_story_tensor, dim=0)

        story_tensors_list = [s.story_tensor for s in child_list]
        child_story_tensors = torch.stack(story_tensors_list)

        for i, c in enumerate(child_list):
            if c.gold == True:
                gold_index = i

        # print(parent_story_tensor.shape, child_story_tensors.shape)

        if self._model._story_feedforward is not None:
            parent_story_tensor = self._model._story_feedforward(parent_story_tensor)

        if self._model._target_feedforward is not None:
            child_story_tensors = self._model._target_feedforward(child_story_tensors)

        logits = self._model.calculate_logits(parent_story_tensor.to(self._device),
                                              child_story_tensors.to(self._device))
        probs = self._softmax(logits)
        log_probs = torch.log(probs)

        logits = torch.squeeze(logits)
        probs = torch.squeeze(probs)
        log_probs = torch.squeeze(log_probs)

        for c, l, p, lb in zip(child_list, logits, probs, log_probs):
            c.logit = l
            c.prob = p
            c.log_prob = lb

        root.logits = logits
        root.probs = probs
        root.log_probs = log_probs

        root.gold_index = gold_index

        self._calculate_distances(root, parent_story_tensor, child_story_tensors, gold_index)

        # print(RenderTree(root))

        return root

    def _calculate_distances(self, root, parent_story_tensor, child_story_tensors, gold_index):
        ''' Compute 1 level differences. May need to change if multiple skip steps are added.
        '''

        # print("Gold Index", gold_index)

        gold_child_tensor = child_story_tensors[gold_index]

        gold_child_tensor = gold_child_tensor.unsqueeze(dim=0).expand_as(child_story_tensors)

        root.surprise_distances_l1 = self._l1(gold_child_tensor, child_story_tensors)
        root.surprise_distances_l1[gold_index] = 0.0
        # print("Surprise L1", root.surprise_distances_l1)
        root.surprise_distances_l2 = self._l2(gold_child_tensor, child_story_tensors)
        root.surprise_distances_l2[gold_index] = 0.0
        # print("Surprise L2", root.surprise_distances_l2)

        parent_story_tensor = parent_story_tensor.expand_as(child_story_tensors)
        root.suspense_distances_l1 = self._l1(parent_story_tensor, child_story_tensors)
        root.suspense_distances_l1[gold_index] = 0.0
        # print("Suspense L1", root.suspense_distances_l1)
        root.suspense_distances_l2 = self._l2(parent_story_tensor, child_story_tensors)
        root.suspense_distances_l2[gold_index] = 0.0
        #print("Suspense L2", root.suspense_distances_l2)


    def _sample_tree(self, instance, outputs):
        ''' Create a hierarchy of possibilities by generating sentences in a tree structure.
        '''
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
                                   sentence_ids=text_to_gen_from,
                                   sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                  text_to_gen_from if t in self.indexer.decoder],
                                   sentence_length=sentence_length, parent=paths)

            correct_futures = []

            # Put the correct representations into the sentences at the given point.
            for i in range(1, self.sentences_to_rollout + 1):

                if position + i > len(text) - 1 or len(text[position + i]) == 0:
                    continue

                text_tensor_future_dict = text[position + i].as_tensor(text_field.get_padding_lengths())
                text_to_gen_from_future = text_tensor_future_dict['openai_transformer']

                sentence_length_future = torch.sum(embedded_text_mask[position + i]).item()
                text_to_gen_from_future = text_to_gen_from_future[0:sentence_length_future]

                correct_future = AnyNode(gold=True,
                                         story_tensor=encoded_stories[position + i].cpu().detach(),
                                         sentence_ids=text_to_gen_from_future,
                                         sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                        text_to_gen_from_future if
                                                        t in self.indexer.decoder],
                                         sentence_length=sentence_length_future)
                correct_futures.append(correct_future)
                # print(f"Add Node: {correct_base}")

            print(len(correct_futures))

            correct_futures = copy.copy(correct_futures)
            future = correct_futures.pop()
            future.parent = correct_base

            for sample_num in range(self.samples_per_level):

                self.generate_sentence(position, embedded_text_tensor, embedded_text_mask, encoded_story,
                                       text_to_gen_from,
                                       sentence_length, ctx_size, recursion=self.sentences_to_rollout - 1,
                                       parent=correct_base, correct_futures=correct_futures)

            # print(RenderTree(root))
            #self._calculate_disc_probabilities(correct_base)
        return root

    def generate_sentence(self, position, embedded_text_tensor, embedded_text_mask, encoded_story,
                          text_to_gen_from, sentence_length, ctx_size, recursion=0, parent=None, correct_futures=None):

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

                encoded_text_merged = self.embedder(torch.unsqueeze(text_tensor_merged.to(self._device), dim=0).long())

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

        # Add newly generated context to sentences.
        gen_sentence_padded = torch.tensor(gen_sentence_padded).long()
        gen_sentence_padded = torch.unsqueeze(gen_sentence_padded, dim=0)

        generated_sentence_tensor = self.embedder(gen_sentence_padded.to(self._device))

        new_embedded_text_tensor = torch.zeros(embedded_text_tensor.shape[0], max(embedded_text_tensor.shape[1],
                                                                                  generated_sentence_tensor.shape[1]),
                                               embedded_text_tensor.shape[2])
        new_embedded_text_tensor[:, 0 : embedded_text_tensor.shape[1], :] = embedded_text_tensor
        new_embedded_text_tensor[position:, 0: generated_sentence_tensor.shape[1], :] = generated_sentence_tensor
        embedded_text_tensor = new_embedded_text_tensor

        new_embedded_text_mask = torch.zeros(embedded_text_tensor.shape[0], embedded_text_tensor.shape[1])
        new_embedded_text_mask[: , 0: embedded_text_mask.shape[1]] = embedded_text_mask
        embedded_text_mask = new_embedded_text_mask.long()

        # Encode as a story.
        encoded_sentences, encoded_story, story_sentence_mask = self._model.encode_story_vectors(
            embedded_text_tensor.to(self._device), embedded_text_mask.to(self._device))
        encoded_story = encoded_story[position]

        encoded_story = torch.squeeze(encoded_story, dim=0)

        encoded_story = torch.squeeze(encoded_story)

        created_node = AnyNode(gold=False, story_tensor=encoded_story.cpu().detach(), sentence_ids=gen_sentence,
                               sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                              gen_sentence if t in self.indexer.decoder],
                               sentence_length=len(gen_sentence),
                               parent=parent)

        #print(created_node)

        if recursion > 0:

            correct_futures = copy.copy(correct_futures)
            future = correct_futures.pop()
            future.parent = created_node

            for i in range(self.samples_per_level):

                next_position = position + 1

                # Don't project past the end of the story.
                if next_position == embedded_text_tensor.shape[0]:
                    break

                self.generate_sentence(next_position, embedded_text_tensor, embedded_text_mask, encoded_story,
                                       torch.tensor(gen_sentence).long(), len(gen_sentence), ctx_size,
                                       recursion=recursion - 1, parent=created_node, correct_futures=correct_futures)

            self._calculate_disc_probabilities(created_node)

    def predict(self, embedded_text, story, sentence_length, ctx_size):

        story = torch.unsqueeze(story, dim=0)
        logits = self._model._lm_model(embedded_text.to(self._device), story.to(self._device))
        logits =  torch.squeeze(logits, dim=0)
        logits = logits[min(sentence_length - 1, ctx_size - 1), :]
        next_word_id = random_sample(logits)

        return next_word_id

