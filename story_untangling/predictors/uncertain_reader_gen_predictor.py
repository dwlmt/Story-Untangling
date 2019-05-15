import copy
from collections import OrderedDict

import more_itertools
import numpy
import scipy
import torch
from allennlp.common import JsonDict
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from anytree import AnyNode, Node, PreOrderIter, LevelOrderGroupIter
from anytree.exporter import DictExporter
from torch.distributions import Categorical
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


def only_probability_nodes(node):
    if isinstance(node, AnyNode) and hasattr(node, "prob"):
        return True
    else:
        return False


def only_state_nodes(node):
    if isinstance(node, AnyNode) and hasattr(node, "chain_prob") and hasattr(node, "suspense_distance_l2"):
        return True
    else:
        return False

def probability_and_tensor(node):
    if only_probability_nodes(node) and only_tensor_nodes():
        return True
    else:
        return False

def only_position_nodes(node):
    if isinstance(node, AnyNode) and hasattr(node, "sentence_id") and hasattr(node, "suspense_l2"):
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

        self.levels_to_rollout = 2
        self.generate_per_level = 5
        self.prob_threshold = 0.02

        self._model.full_output_embedding = True
        self._model.run_feedforwards = False  # Turn off normal feedforwards to avoid running twice.

        self._sliding_windows = [3, 5, 7]

        self.embedder = model._text_field_embedder._token_embedders["openai_transformer"]
        self.embedder._top_layer_only = True

        self.tokenizer = dataset_reader._tokenizer
        self.indexer =  dataset_reader._token_indexers["openai_transformer"]

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
                    if len(value.shape) == 0:
                        n.__dict__[attr] = value.item()
                    else:
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

        child_story_tensors, parent_story_tensor = self._run_feedforward(child_story_tensors, parent_story_tensor)

        logits = self._model.calculate_logits(parent_story_tensor.to(self._device),
                                              child_story_tensors.to(self._device))
        probs = self._softmax(logits)
        log_probs = torch.log(probs)

        logits = torch.squeeze(logits)
        probs = torch.squeeze(probs)
        log_probs = torch.squeeze(log_probs)

        for i, (c, l, p, lb) in enumerate(zip(child_list, logits, probs, log_probs)):
            c.logit = l
            c.prob = p
            c.log_prob = lb

            # Detach all child nodes that don't meet the probability threshold.
            if c.gold:
                pass
            elif c.prob < self.prob_threshold:
                print(f"Remove low probability continuation, prob {c.prob}, {c.sentence_text}")
                c.parent = None
                root.children = root.children[: i] + root.children[i + 1:]
            else:
                pass  # print(f"Include, prob {c.prob}, {c.sentence_text}")


        root.gold_index = gold_index

        self._calculate_distances(root, parent_story_tensor, child_story_tensors, gold_index)

        return root

    def _run_feedforward(self, child_story_tensors, parent_story_tensor):
        if self._model._story_feedforward is not None:
            parent_story_tensor = self._model._story_feedforward(parent_story_tensor)
        if self._model._target_feedforward is not None:
            child_story_tensors = self._model._target_feedforward(child_story_tensors)
        return child_story_tensors, parent_story_tensor

    def _calculate_distances(self, root, parent_story_tensor, child_story_tensors, gold_index):
        ''' Compute 1 level differences. May need to change if multiple skip steps are added.
        '''

        gold_child_tensor = child_story_tensors[gold_index]

        gold_child_tensor = gold_child_tensor.unsqueeze(dim=0).expand_as(child_story_tensors)

        surprise_distances_l1 = self._l1(gold_child_tensor, child_story_tensors)
        surprise_distances_l2 = self._l2(gold_child_tensor, child_story_tensors)

        parent_story_tensor = parent_story_tensor.expand_as(child_story_tensors)
        suspense_distances_l1 = self._l1(parent_story_tensor, child_story_tensors)
        suspense_distances_l2 = self._l2(parent_story_tensor, child_story_tensors)

        for i, c in enumerate(root.children):

            if not c.gold:
                c.surprise_distance_l1 = surprise_distances_l1[i]
                c.surprise_distance_l2 = surprise_distances_l2[i]

                c.suspense_distance_l1 = suspense_distances_l1[i]
                c.suspense_distance_l2 = suspense_distances_l2[i]

    def _sample_tree(self, instance, outputs):
        ''' Create a hierarchy of possibilities by generating sentences in a tree structure.
        '''
        embedded_text_tensor = outputs["embedded_text_tensor"]
        embedded_text_tensor = torch.from_numpy(embedded_text_tensor).cpu()
        embedded_text_mask = torch.from_numpy(outputs["masks"]).cpu()
        encoded_stories = torch.from_numpy(outputs["source_encoded_stories"]).cpu()
        text = instance["text"]
        root = Node(name="root")
        per_sentence = Node(name="position", parent=root)
        # aggregate = Node(name="aggregate_stats", parent=root)
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

            correct_futures = []

            # Put the correct representations into the sentences at the given point.
            parent_story_tensor = None
            for i in range(1, self.levels_to_rollout + 2):

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

                if parent_story_tensor is not None:
                    child_story_tensor_fwd, parent_story_tensor_fwd = self._run_feedforward(
                        encoded_stories[position + i],
                        parent_story_tensor)
                    child_story_tensor_fwd = torch.unsqueeze(child_story_tensor_fwd, dim=0)
                    correct_future.suspense_distance_l1 = torch.squeeze(
                        self._l1(child_story_tensor_fwd, parent_story_tensor_fwd), dim=0)
                    correct_future.suspense_distance_l2 = torch.squeeze(
                        self._l2(child_story_tensor_fwd, parent_story_tensor_fwd), dim=0)

                correct_futures.append(correct_future)
                parent_story_tensor = torch.unsqueeze(encoded_stories[position + i], dim=0)

            if len(correct_futures) > 0:

                correct_futures_copy = copy.deepcopy(correct_futures)
                base = correct_futures_copy.pop()
                base.parent = position_node

                for sample_num in range(self.generate_per_level):
                    self.generate_sentence(position, embedded_text_tensor, embedded_text_mask, encoded_story,
                                           text_to_gen_from,
                                           sentence_length, ctx_size, recursion=self.levels_to_rollout - 1,
                                           parent=base, correct_futures=correct_futures)

                self._calculate_disc_probabilities(base)

                self._calc_chain_probs(base)

                # Calculate suspense across the tree and merge into the base.
                metrics = self._calc_state_based_suspense(base)
                position_node.__dict__ = {**position_node.__dict__, **metrics}

        self._calc_summary_stats(root)

        return root

    def _calc_summary_stats(self, root):

        stats_node = Node(name="batch_stats", parent=root)
        stats_source_dict = {"suspense_l1": [], "suspense_l2": [],
                             "surprise_l1": [], "surprise_l2": [], "suspense_entropy": []}

        for n in PreOrderIter(root, only_position_nodes):
            for k, v in stats_source_dict.items():
                stat = n.__dict__[k]
                stats_source_dict[k].append(stat)

        stats_dict = {}
        for k, v in stats_source_dict.items():
            v_array = numpy.asarray(v)
            nobs, minmax, mean, variance, skewness, kurtosis = scipy.stats.describe(v_array)

            stats_dict["num"] = nobs
            stats_dict["min"] = minmax[0]
            stats_dict["max"] = minmax[1]
            stats_dict["variance"] = variance
            stats_dict["std"] = variance ** (.5)
            stats_dict["skew"] = skewness
            stats_dict["kurtosis"] = kurtosis

            for p in [25, 50, 75]:
                stats_dict[f"{p}_perc"] = numpy.percentile(v_array, p)

            AnyNode(parent=stats_node, name=f"{k}", **stats_dict)

            window_stats_node = Node(name="window_stats", parent=root)

            for n in self._sliding_windows:
                window_node = Node(name=f"{n}", parent=window_stats_node)
                windows = more_itertools.windowed(v, n)
                for i, window in enumerate(windows):
                    window = [w for w in window if w is not None]

                    if len(window) == 0:
                        continue

                    v_array = numpy.asarray(window)
                    mean = numpy.mean(v_array)
                    median = numpy.median(v_array)
                    variance = numpy.var(v_array)
                    std = variance ** (.5)
                    AnyNode(parent=window_node, position=i, mean=mean, median=median, variance=variance, std=std)

    def _calc_state_based_suspense(self, root):
        # Use to keep results
        metrics_dict = {}

        suspense_l1_culm = 0.0
        suspense_l2_culm = 0.0
        surprise_l1_culm = 0.0
        surprise_l2_culm = 0.0

        steps_counter = 0

        # Iterate and add chain probabilities through the tree structure.
        for i, node_group in enumerate(LevelOrderGroupIter(root, only_state_nodes), start=0):

            # Don't calculate on the base root.
            if len(node_group) <= 1:
                continue

            steps_counter += 1

            probs = torch.stack([n.chain_prob for n in node_group]).to(self._device)

            distribution = Categorical(probs)

            entropy = distribution.entropy().cpu().item()
            metrics_dict[f"suspense_entropy_{i}"] = entropy

            suspense_l1 = torch.stack([n.suspense_distance_l1 for n in node_group]).to(self._device)
            suspense_l1_stat = torch.squeeze(torch.sum((suspense_l1 * probs), dim=-1)).cpu().item()
            suspense_l1_culm += suspense_l1_stat
            metrics_dict[f"suspense_l1_{i}"] = suspense_l1_stat

            suspense_l2 = torch.stack([n.suspense_distance_l2 for n in node_group]).to(self._device)
            suspense_l2_stat = torch.squeeze(torch.sum((suspense_l2 * probs), dim=-1)).cpu().item()
            suspense_l2_culm += suspense_l2_stat
            metrics_dict[f"suspense_l2_{i}"] = suspense_l2_stat

            # Exclude gold for surprise as already calculated in the distances.
            nodes_exclude_gold = [n for n in node_group if n.gold == False]

            surprise_l1_stat = torch.squeeze(
                torch.sum((torch.stack([n.surprise_distance_l1 for n in nodes_exclude_gold]).to(self._device)),
                          dim=-1)).cpu().item()
            surprise_l1_culm += surprise_l1_stat
            metrics_dict[f"surprise_l1_{i}"] = surprise_l1_stat

            surprise_l2_stat = torch.squeeze(
                torch.sum((torch.stack([n.surprise_distance_l2 for n in nodes_exclude_gold]).to(self._device)),
                          dim=-1)).cpu().item()
            surprise_l2_culm += surprise_l2_stat
            metrics_dict[f"surprise_l2_{i}"] = surprise_l2_stat

        # Use the last value for the main suspense.
        metrics_dict["suspense_entropy"] = entropy

        metrics_dict["suspense_l1"] = suspense_l1_culm
        metrics_dict["suspense_l2"] = suspense_l2_culm

        metrics_dict["surprise_l1"] = suspense_l1_culm
        metrics_dict["surprise_l2"] = suspense_l2_culm

        metrics_dict["steps"] = steps_counter
        return metrics_dict

    def _calc_chain_probs(self, root):
        ''' Recusrively calculate chain probabilities down the network.
        '''

        # Iterate and add chain probabilities through the tree structure.
        for node in PreOrderIter(root, only_probability_nodes):

            node.chain_prob = node.prob
            node.chain_log_prob = node.log_prob

            if node.parent is not None and hasattr(node.parent, "chain_prob"):
                node.chain_prob *= node.parent.chain_prob
                node.chain_log_prob += node.parent.chain_log_prob

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

        if recursion > 0:

            correct_futures_copy = copy.deepcopy(correct_futures)
            future = correct_futures_copy.pop()
            future.parent = created_node

            for i in range(self.generate_per_level):

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

