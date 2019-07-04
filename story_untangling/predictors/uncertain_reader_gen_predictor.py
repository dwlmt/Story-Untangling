import copy
from collections import OrderedDict

import more_itertools
import numpy
import pandas
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

def random_sample(logits: torch.Tensor, top_k_words=None) -> int:
    indices = None
    if top_k_words is not None and top_k_words > 0:
        logits, indices = torch.topk(logits, k=top_k_words)

    d = torch.distributions.Categorical(logits=logits)
    sampled = d.sample()

    if indices is not None:
        sampled = indices[sampled]

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
    if isinstance(node, AnyNode) and hasattr(node, "chain_prob") and hasattr(node, "parent_distance_l2"):
        return True
    else:
        return False

def probability_and_tensor(node):
    if only_probability_nodes(node) and only_tensor_nodes():
        return True
    else:
        return False

def only_position_nodes(node):
    if isinstance(node, AnyNode) and hasattr(node, "sentence_id") and (
            hasattr(node, "corpus_suspense_l2") or hasattr(node, "generated_suspense_l2")):
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

        story_id_file = "/afs/inf.ed.ac.uk/group/project/comics/stories/WritingPrompts/annotation_results/raw/story_id_mapping.csv"

        story_id_df = pandas.read_csv(story_id_file)
        self.story_ids_to_predict = set(story_id_df['story_id'])
        self.only_annotation_stories = False

        self.levels_to_rollout = 1
        self.generate_per_branch = 25
        self.sample_per_level_branch = 25

        self.max_leaves_to_keep_per_branch = 0
        self.probability_mass_to_keep_per_branch = 0.0

        self.global_beam_size = 10

        self.sample_top_k_words = 50

        self.min_sentence_length = 3
        self.max_generated_sentence_length  = 150
        self.context_sentence_to_generate_from = 8

        self._remove_sentence_output = True

        self._model.full_output_embedding = True
        self._model.run_feedforwards = False  # Turn off normal feedforwards to avoid running twice.

        self._sliding_windows = [1, 3, 5, 7]
        self._generation_sampling_temperature = 1.0
        self._discrimination_temperature = 1.0
        self._cosine = True

        self.embedder = model._text_field_embedder._token_embedders["openai_transformer"]
        self.embedder._top_layer_only = True

        self.dataset_reader = dataset_reader
        self.dataset_reader._story_chunking = 150  # Allow bigger batching for sampling.
        self.tokenizer = dataset_reader._tokenizer
        self.indexer =  dataset_reader._token_indexers["openai_transformer"]

        self.keep_tensor_output = False

        self._device = self._model._lm_model._decoder.weight.device

        self._softmax = Softmax(dim=-1)

        self._l1 = PairwiseDistance(p=1.0)
        self._l2 = PairwiseDistance(p=2.0)


    def predict_instance(self, instance: Instance) -> JsonDict:

        story_id = instance["metadata"]["story_id"]
        if story_id not in self.story_ids_to_predict and self.only_annotation_stories:
            print(f"Skipping non annotation story: {story_id}")
            return None

        print(f"Predicting story: {story_id}")
        print(f'{instance["metadata"]["text"]}')

        outputs = self._model.forward_on_instance(instance)

        root = self._sample_tree(instance, outputs)

        if self._remove_sentence_output:
            for n in PreOrderIter(root, only_position_nodes):
                n.children = ()

        for n in PreOrderIter(root, only_tensor_nodes):

            for attr, value in n.__dict__.items():
                if isinstance(value, torch.Tensor):
                    if len(value.shape) == 0:
                        n.__dict__[attr] = value.item()
                    else:
                        n.__dict__[attr] = value.tolist()

            if not self.keep_tensor_output:
                n.story_tensor = None
                n.sentence_tensor = None
                n.embedded_text_tensor = None
                n.embedded_text_mask = None
                n.indexed_tokens = None

        root_as_dict = exporter.export(root)
        print(root_as_dict)
        return sanitize(root_as_dict)

    def _calculate_disc_probabilities(self, root):
        ''' Iterate over the tree and calculate the probabilities for each path using the discriminate loss function.
        '''
        child_list = []

        for child in PreOrderIter(root, only_tensor_nodes, maxlevel=2):
            if child.parent == root:
                child_list.append(child)

        if len(child_list) == 0:
            return

        child_story_tensors, log_probs, logits, parent_story_tensor, probs = self.calculate_embedding_probs_and_logits(child_list, root)

        cutoff_prob = self.calc_probability_cutoff(probs)

        pruned_child_list = []
        for i, (c, l, p, lb) in enumerate(zip(child_list, logits, probs, log_probs)):
            c.logit = l
            c.prob = p
            c.log_prob = lb

            calc_prob = c.prob

            if c.gold:
                pass
            elif calc_prob < cutoff_prob:
                c.parent = None
                root.children = root.children[: i] + root.children[i + 1:]
            else:

                pruned_child_list.append(c)

        self._calculate_distances(child_list, parent_story_tensor, child_story_tensors)

        root.children = child_list

        return root

    def calc_probability_cutoff(self, probs):
        # Set the probability to prune unlikely nodes.
        if probs is None or len(probs.shape) == 0:
            return

        cutoff_prob = 1.0
        probs_from_max_to_min = sorted(probs, reverse=True)

        if self.max_leaves_to_keep_per_branch > 0:
            top_n_prob_cutoff = 1.0
            if len(probs_from_max_to_min) > self.max_leaves_to_keep_per_branch:
                top_n_prob_cutoff = probs_from_max_to_min[self.max_leaves_to_keep_per_branch - 1]

                cutoff_prob = min(cutoff_prob, top_n_prob_cutoff)

        if self.probability_mass_to_keep_per_branch > 0.0:
            probability_mass_cutoff = 0.0
            culm = 0.0
            for p in probs_from_max_to_min:

                culm += p

                if culm > self.probability_mass_to_keep_per_branch:
                    break

                probability_mass_cutoff = p

            cutoff_prob = min(cutoff_prob, probability_mass_cutoff)


        return cutoff_prob

    def calculate_embedding_probs_and_logits(self, child_list, root):
        parent_story_tensor = root.story_tensor
        parent_story_tensor = torch.unsqueeze(parent_story_tensor, dim=0).to(self._device)
        story_tensors_list = [s.story_tensor for s in child_list]
        child_story_tensors = torch.stack(story_tensors_list).to(self._device)


        if not self._cosine:
            parent_story_tensor_fwd = parent_story_tensor
            child_story_tensors_fwd = child_story_tensors
        else:
            parent_story_tensor_fwd = torch.norm(parent_story_tensor, p=2, dim=-1, keepdim=True)
            child_story_tensors_fwd  = torch.norm(child_story_tensors, p=2, dim=-1, keepdim=True)

        child_story_tensors, parent_story_tensor = self._run_feedforward(child_story_tensors, parent_story_tensor)
        logits = self._model.calculate_logits(parent_story_tensor_fwd,
                                              child_story_tensors_fwd)

        probs = self._softmax(logits / self._discrimination_temperature)

        log_probs = torch.log(probs)
        logits = torch.squeeze(logits)

        probs = torch.squeeze(probs)
        log_probs = torch.squeeze(log_probs)
        return child_story_tensors, log_probs, logits, parent_story_tensor, probs

    def _run_feedforward(self, child_story_tensors, parent_story_tensor):
        if self._model._story_feedforward is not None:
            parent_story_tensor = self._model._story_feedforward(parent_story_tensor)
        if self._model._target_feedforward is not None:
            child_story_tensors = self._model._target_feedforward(child_story_tensors)
        return child_story_tensors, parent_story_tensor

    def _calculate_distances(self, children, parent_story_tensor, child_story_tensors):
        ''' Compute 1 level differences. May need to change if multiple skip steps are added.
        '''

        parent_story_tensor = parent_story_tensor.expand_as(child_story_tensors)
        parent_distances_l1 = self._l1(parent_story_tensor, child_story_tensors)
        parent_distances_l2 = self._l2(parent_story_tensor, child_story_tensors)

        for i, c in enumerate(children):
            c.parent_distance_l1 = parent_distances_l1[i]
            c.parent_distance_l2 = parent_distances_l2[i]

    def _sample_tree(self, instance, outputs):
        ''' Create a hierarchy of possibilities by generating sentences in a tree structure.
        '''
        embedded_text_tensor = outputs["embedded_text_tensor"]
        embedded_text_tensor = torch.from_numpy(embedded_text_tensor).cpu()
        embedded_text_mask = torch.from_numpy(outputs["masks"]).cpu().long()
        encoded_story = torch.from_numpy(outputs["source_encoded_stories"]).cpu()
        indexed_tokens = torch.from_numpy(outputs["indexed_tokens"]).cpu()
        text = instance["text"]
        root = Node(name="predictions")
        per_sentence = Node(name="position", parent=root)

        for position, text_field in enumerate(text):

            if position == embedded_text_tensor.shape[0]:
                break

            sentence_length = int(torch.sum(embedded_text_mask[position]).item())

            if sentence_length == 0:
                continue

            story_id = instance["metadata"]["story_id"]
            sentence_ids = instance["metadata"]["sentence_ids"]
            sentence_nums = instance["metadata"]["sentence_nums"]
            story_text = instance["metadata"]["text"]

            if position >= len(sentence_ids):
                continue

            sentence_id = sentence_ids[position]
            sentence_num = sentence_nums[position]
            sentence_text = story_text[position]

            position_node = AnyNode(name=f"{position}", story_id=story_id,
                                    sentence_id=sentence_id,
                                    sentence_num=sentence_num,
                                    sentence_text=sentence_text,
                                    parent=per_sentence)

            correct_futures = self.create_correct_futures(embedded_text_mask, encoded_story, position, text,
                                                          text_field, story_id, sentence_ids, sentence_nums)



            if len(correct_futures) > 0:

                generated = AnyNode(name="generated", parent=position_node)
                base_generated = self.generated_expansion(generated, copy.deepcopy(correct_futures),
                                                          embedded_text_mask,
                                                          embedded_text_tensor,
                                                          encoded_story,
                                                          indexed_tokens,
                                                          position)

                corpus = AnyNode(name="corpus", parent=position_node)
                base_corpus = self.sampling_expansion(corpus, copy.deepcopy(correct_futures), embedded_text_mask,
                                                      embedded_text_tensor, position)


                # Calculate suspense across the tree and merge into the base.
                if self.generate_per_branch > 0:
                    metrics = self._calc_state_based_suspense(base_generated, type="generated")
                    position_node.__dict__ = {**position_node.__dict__, **metrics}

                if self.sample_per_level_branch > 0:
                    metrics = self._calc_state_based_suspense(base_corpus, type="corpus")
                    position_node.__dict__ = {**position_node.__dict__, **metrics}


        self._calc_summary_stats(root)

        return root

    def generated_expansion(self, type_node, correct_futures, embedded_text_mask, embedded_text_tensor,
                            encoded_story, indexed_tokens, position):

        if self.generate_per_branch > 0:

            base = correct_futures.pop(0)
            gold = copy.deepcopy(base)
            gold.parent = type_node

            gold.embedded_text_mask = embedded_text_mask.cpu()
            gold.embedded_text_tensor = embedded_text_tensor.cpu()
            gold.indexed_tokens = indexed_tokens.cpu()

            base_correct = gold
            parents = [gold]

            for i in range(1, self.levels_to_rollout + 1):

                if len(correct_futures) == 0:
                    continue

                new_parents = []

                for parent in parents:

                    print("Generate from:", parent.sentence_text)

                    if len(parent.children) > 0:
                        children = list(parent.children)
                    else:
                        children = []

                    for j in range(self.generate_per_branch):

                        generated_sentence = self.generate_sentence(position, indexed_tokens, encoded_story)

                        created_node = self.encode_sentence_node(generated_sentence, position, i,
                                                                              parent.embedded_text_tensor.clone().to(
                                                                                  self._device),
                                                                              parent.embedded_text_mask.clone().to(
                                                                                  self._device),
                                                                              parent.indexed_tokens.clone().to(
                                                                                  self._device),
                                                                              parent=parent)
                        if created_node:
                            children.append(created_node)

                            print("Generated Distance", self._l2(torch.unsqueeze(parent.story_tensor, dim=0),
                                                               torch.unsqueeze(created_node.story_tensor, dim=0)),
                                  parent.sentence_text, created_node.sentence_text)

                    if parent.gold and len(children) > 0:
                        future_gold = correct_futures.pop(0)
                        future_gold.parent = gold
                        gold = future_gold

                        gold.embedded_text_mask = parent.embedded_text_mask.cpu()
                        gold.embedded_text_tensor = parent.embedded_text_tensor.cpu()
                        gold.indexed_tokens = parent.indexed_tokens

                        print("Gold Distance Tensor", self._l2(torch.unsqueeze(parent.story_tensor, dim=0),
                                                               torch.unsqueeze(gold.story_tensor, dim=0)),
                              parent.sentence_text, gold.sentence_text)

                        children.append(gold)

                    parent.children = children

                    # Greater than 1 as one child is the propagated gold.

                    self._calculate_disc_probabilities(parent)
                    self._calc_chain_probs(parent)

                new_parents.extend(list(parent.children))

                if len(new_parents) > 1:
                    new_parents = sorted(new_parents, key=lambda p: (p.gold, p.chain_prob), reverse=True)

                if len(new_parents) > self.global_beam_size - 1 and self.global_beam_size > 0:
                    new_parents = new_parents[0:self.global_beam_size - 1]

                parents = new_parents

                print("Retained: ", [(p.chain_prob, p.prob, p.gold, p.sentence_text) for p in parents])

            return base_correct

    def sampling_expansion(self, type_node, correct_futures, embedded_text_mask, embedded_text_tensor,
                           position):

        if self.sample_per_level_branch > 0:

            #print(correct_futures)
            base = correct_futures.pop(0)
            gold = copy.deepcopy(base)
            gold.parent = type_node

            gold.embedded_text_mask = embedded_text_mask.cpu()
            gold.embedded_text_tensor = embedded_text_tensor.cpu()
            
            base_correct = gold
            parents = [gold]

            for i in range(1, self.levels_to_rollout + 1):

                if len(correct_futures) == 0:
                    continue

                new_parents = []

                for parent in parents:

                    print("Sample from:", parent.sentence_text)

                    if len(parent.children) > 0:
                        children = list(parent.children)
                    else:
                        children = []
                    parent.children = children

                    for j in range(self.sample_per_level_branch):

                        created_node = self.sample_sentences_from_corpus(position, i,
                                                                         parent.embedded_text_tensor.clone().to(
                                                                             self._device),
                                                                         parent.embedded_text_mask.clone().to(
                                                                             self._device),
                                                                         parent=parent)
                        if created_node:
                            children.append(created_node)

                            print("Sampled Distance", self._l2(torch.unsqueeze(parent.story_tensor, dim=0),
                                                               torch.unsqueeze(created_node.story_tensor, dim=0)),
                                  parent.sentence_text, created_node.sentence_text)

                    if parent.gold and len(children) > 0:
                        future_gold = correct_futures.pop(0)
                        future_gold.parent = gold
                        gold = future_gold

                        gold.embedded_text_mask = parent.embedded_text_mask.cpu()
                        gold.embedded_text_tensor = parent.embedded_text_tensor.cpu()

                        print("Gold Distance Tensor", self._l2(torch.unsqueeze(parent.story_tensor, dim=0),
                                                               torch.unsqueeze(gold.story_tensor, dim=0)),
                              parent.sentence_text, gold.sentence_text)

                        children.append(gold)

                    parent.children = children

                    self._calculate_disc_probabilities(parent)
                    self._calc_chain_probs(parent)

                    new_parents.extend(list(parent.children))

                if len(new_parents) > 1:
                    new_parents = sorted(new_parents, key=lambda p: (p.gold, p.chain_prob), reverse=True)

                if len(new_parents) > self.global_beam_size - 1 and self.global_beam_size > 0:
                    new_parents = new_parents[0:self.global_beam_size - 1]

                parents = new_parents

                print("Retained: ", [(p.chain_prob, p.prob, p.gold, p.sentence_text) for p in parents])

            return base_correct

    def create_correct_futures(self, embedded_text_mask, encoded_stories, position, text, text_field, story_id,
                               sentence_ids, sentence_nums):
        correct_futures = []
        # Put the correct representations into the sentences at the given point.
        parent_story_tensor = None
        for i in range(0, self.levels_to_rollout + 2):

            if position + i > len(text) - 1 or len(text[position + i]) == 0:
                continue

            text_tensor_future_dict = text[position + i].as_tensor(text_field.get_padding_lengths())
            text_to_gen_from_future = text_tensor_future_dict['openai_transformer']

            sentence_length_future = int(torch.sum(embedded_text_mask[position + i]).item())
            text_to_gen_from_future = text_to_gen_from_future[0:sentence_length_future]

            text_as_list = text_to_gen_from_future.tolist()

            correct_future = AnyNode(gold=True,
                                     story_tensor=encoded_stories[position + i].cpu().detach(),
                                     token_ids=text_to_gen_from_future,
                                     story_id=story_id,
                                     sentence_id=sentence_ids[position + i],
                                     sentence_num=sentence_nums[position + i],
                                     sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                    text_as_list if
                                                    t in self.indexer.decoder],
                                     sentence_length=len(text_as_list))

            if parent_story_tensor is not None:
                child_story_tensor_fwd, parent_story_tensor_fwd = self._run_feedforward(
                    encoded_stories[position + i],
                    parent_story_tensor)
                child_story_tensor_fwd = torch.unsqueeze(child_story_tensor_fwd, dim=0)
                correct_future.parent_distance_l1 = torch.squeeze(
                    self._l1(child_story_tensor_fwd, parent_story_tensor_fwd), dim=0)
                correct_future.parent_distance_l2 = torch.squeeze(
                    self._l2(child_story_tensor_fwd, parent_story_tensor_fwd), dim=0)

            correct_futures.append(correct_future)
            parent_story_tensor = torch.unsqueeze(encoded_stories[position + i], dim=0)
        return correct_futures

    def sample_sentences_from_corpus(self, position, level, embedded_text_tensor, embedded_text_mask,
                                     parent):

        story_id = parent.story_id
        sentence_id = parent.sentence_id

        instance = self.dataset_reader.sample_random_sentences(self.sample_per_level_branch, story_id=story_id,
                                                                   sentence_id=sentence_id)

        field = instance["text"]
        metadata = instance["metadata"]

        field.index(self._model._vocab)
        padding_lengths = field.get_padding_lengths()
        tensor_dict = field.as_tensor(padding_lengths)

        indexed_tokens = tensor_dict["openai_transformer"].to(self._device)
        non_empty_mask = torch.sum(indexed_tokens, dim=1) > 0

        indexed_tokens = indexed_tokens[non_empty_mask, :]

        encoded_text_merged = self.embedder(indexed_tokens.to(self._device))

        for i, (gen_sentence, story_id, sentence_id, sentence_num, indexed) in enumerate(
                zip(torch.split(encoded_text_merged, 1),
                    metadata["story_id"],
                    metadata["sentence_ids"],
                    metadata["sentence_nums"],
                    torch.split(indexed_tokens, 1))):

            encoded_story, encoded_sentence, embedded_text_tensor, embedded_text_mask = self.encode_story(
                embedded_text_mask.to(self._device),
                embedded_text_tensor.to(self._device),
                gen_sentence.to(self._device),
                torch.squeeze(indexed.to(self._device), dim=0),
                position + level)

            encoded_story, encoded_story.cpu()

            token_ids = [t for t in indexed_tokens[i].tolist() if t != 0]

            sentence_length = len(token_ids)
            if not sentence_length < self.min_sentence_length:
                created_node = AnyNode(gold=False, story_tensor=encoded_story.cpu().detach(),
                                       sentence_tensor=encoded_sentence.cpu().detach(),
                                       token_ids=token_ids,
                                       level=level,
                                       embedded_text_tensor=embedded_text_tensor.cpu().detach(),
                                       embedded_text_mask=embedded_text_mask.cpu().detach(),
                                       sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                      token_ids if t in self.indexer.decoder],
                                       sentence_length=sentence_length, type="corpus",
                                       story_id=story_id, sentence_id=sentence_id, sentence_num=sentence_num,
                                       parent=parent)

                return created_node
        return None


    def _calc_summary_stats(self, root):

        stats_node = Node(name="batch_stats", parent=root)
        window_stats_node = Node(name="window_stats", parent=root)

        stats_source_dict = {}
        if self.generate_per_branch > 0:
            type = "generated"
            stats_source_dict = {**stats_source_dict, **{f"{type}_suspense_l1": [], f"{type}_suspense_l2": [],
                                                         f"{type}_surprise_l1": [], f"{type}_surprise_l2": [],
                                                         f"{type}_suspense_entropy": []}}

        if self.sample_per_level_branch > 0:
            type = "corpus"
            stats_source_dict = {**stats_source_dict, **{f"{type}_suspense_l1": [], f"{type}_suspense_l2": [],
                                                         f"{type}_surprise_l1": [], f"{type}_surprise_l2": [],
                                                         f"{type}_suspense_entropy": []}}


        story_id = -1
        for n in PreOrderIter(root, only_position_nodes):

            for k, v in stats_source_dict.items():
                if k in n.__dict__:
                    stat = n.__dict__[k]
                    story_id = n.story_id
                    stats_source_dict[k].append(stat)

        stats_dict = {}
        for (k, v) in stats_source_dict.items():

            if v is None or len(v) == 0:
                continue

            v_array = numpy.asarray(v)
            nobs, minmax, mean, variance, skewness, kurtosis = scipy.stats.describe(v_array)

            stats_dict["num"] = nobs
            stats_dict["min"] = minmax[0]
            stats_dict["max"] = minmax[1]
            stats_dict["mean"] = mean
            stats_dict["variance"] = variance
            stats_dict["std"] = variance ** (.5)
            stats_dict["skew"] = skewness
            stats_dict["kurtosis"] = kurtosis
            stats_dict["story_id"] = story_id

            for p in [25, 50, 75]:
                stats_dict[f"{p}_perc"] = numpy.percentile(v_array, p)

            AnyNode(parent=stats_node, name=f"{k}", **stats_dict)

            window_variable_node = Node(name=f"{k}", parent=window_stats_node)

            for n in self._sliding_windows:
                window_node = Node(name=f"{n}", parent=window_variable_node)
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
                    max = numpy.amax(v_array)
                    min = numpy.amax(v_array)
                    AnyNode(parent=window_node, position=i, mean=mean, median=median, variance=variance, std=std,
                            max=max, min=min, story_id=story_id)

    def _calc_state_based_suspense(self, root, type="generated"):
        # Use to keep results
        metrics_dict = {}

        suspense_l1_culm = 0.0
        suspense_l2_culm = 0.0
        surprise_l1_culm = 0.0
        surprise_l2_culm = 0.0

        entropy = 0.0

        steps_counter = 0

        # Iterate and add chain probabilities through the tree structure.
        for i, node_group in enumerate(LevelOrderGroupIter(root, only_state_nodes), start=0):

            # Don't calculate on the base root.
            if len(node_group) <= 2:
                continue


            # Exclude gold for surprise as already calculated in the distances.
            nodes_gold = [n for n in node_group if n.gold == True]

            if len(nodes_gold) > 0:
                gold = nodes_gold[0]


                if hasattr(gold, "parent_distance_l1"):
                    surprise_l1_stat = gold.parent_distance_l1.cpu().item()
                    print("L1 surprise distance", surprise_l1_stat)
                    surprise_l1_culm += surprise_l1_stat
                    metrics_dict[f"{type}_surprise_l1_{i}"] = surprise_l1_stat

                if hasattr(gold, "parent_distance_l2"):
                    surprise_l2_stat = gold.parent_distance_l2.cpu().item()
                    surprise_l2_culm += surprise_l2_stat
                    metrics_dict[f"{type}_surprise_l2_{i}"] = surprise_l2_stat
                    print("L2 surprise distance", surprise_l2_stat)

            steps_counter += 1

            probs = torch.stack([n.chain_prob for n in node_group]).to(self._device)

            distribution = Categorical(probs)

            entropy = distribution.entropy().cpu().item()
            metrics_dict[f"{type}_suspense_entropy_{i}"] = entropy

            parent_l1 = torch.stack([n.parent_distance_l1 for n in node_group]).to(self._device)

            suspense_l1_stat = torch.squeeze(torch.sum((parent_l1 * probs), dim=-1)).cpu().item()
            suspense_l1_culm += suspense_l1_stat
            metrics_dict[f"{type}_suspense_l1_{i}"] = suspense_l1_stat

            parent_l2 = torch.stack([n.parent_distance_l2 for n in node_group]).to(self._device)

            suspense_l2_stat = torch.squeeze(torch.sum((parent_l2 * probs), dim=-1)).cpu().item()
            suspense_l2_culm += suspense_l2_stat
            metrics_dict[f"{type}_suspense_l2_{i}"] = suspense_l2_stat

        # Use the last value for the main suspense.
        metrics_dict[f"{type}_suspense_entropy"] = entropy

        metrics_dict[f"{type}_suspense_l1"] = suspense_l1_culm
        metrics_dict[f"{type}_suspense_l2"] = suspense_l2_culm

        metrics_dict[f"{type}_surprise_l1"] = surprise_l1_culm
        metrics_dict[f"{type}_surprise_l2"] = surprise_l2_culm

        metrics_dict["steps"] = steps_counter
        return metrics_dict

    def _calc_chain_probs(self, root):
        ''' Recursively calculate chain probabilities down the network.
        '''
        # Iterate and add chain probabilities through the tree structure.
        for node in PreOrderIter(root, only_probability_nodes):

            if hasattr(node, "chain_prob"):
                continue  # don't add if it exists.

            node.chain_prob = node.prob
            node.chain_log_prob = node.log_prob

            if node.parent is not None and hasattr(node.parent, "chain_prob"):

                node.chain_prob *= node.parent.chain_prob
                node.chain_log_prob += node.parent.chain_log_prob


    def generate_sentence(self, position, indexed_tokens, encoded_story):

        ctx_size = self.max_generated_sentence_length

        gen_sentence = []

        start_pos = max(0, position - self.context_sentence_to_generate_from)
        end_pos = position + 1

        indexed_tokens_context = indexed_tokens[start_pos : end_pos]
        encoded_story_context = encoded_story[start_pos : end_pos]

        sentence_sums = torch.sum(indexed_tokens_context > 0, dim=1)

        indexed_tokens_context_masked = torch.cat([indexed_tokens_context[x, 0: y ] for x, y in
             enumerate(sentence_sums)], dim=0).to(self._device)

        encoded_story_context_masked = torch.cat([torch.unsqueeze(encoded_story_context[x], dim=0).expand(y, -1) for x, y in enumerate(sentence_sums)]
      , dim=0)

        indexed_tokens_context_merged = indexed_tokens_context_masked
        indexed_length = indexed_tokens_context_masked.shape[0]

        for i in range(ctx_size):
            if len(gen_sentence) > 0:
                indexed_tokens_context_merged = torch.cat((indexed_tokens_context_masked, torch.tensor(gen_sentence).long().to(self._device)))
                indexed_length = indexed_tokens_context_merged.shape[0]

            if indexed_length > ctx_size:

                indexed_tokens_context_merged = indexed_tokens_context_merged[indexed_length - ctx_size:]
                encoded_story_context_masked = encoded_story_context_masked[encoded_story_context_masked.shape[0] - ctx_size:]

            elif indexed_length < ctx_size:
                blank_index = torch.zeros((ctx_size)).long().to(self._device)
                blank_index[0:indexed_tokens_context_merged.shape[0]] = indexed_tokens_context_merged
                indexed_tokens_context_merged = blank_index

            encoded_text_context_masked = self.embedder(torch.unsqueeze(indexed_tokens_context_merged,dim=0))
            encoded_text_context_masked = torch.squeeze(encoded_text_context_masked, dim=0)

            next_word_id = self.predict(encoded_text_context_masked, encoded_story_context_masked)

            if not next_word_id:
                break

            encoded_story_context_masked = torch.cat((encoded_story_context_masked, torch.unsqueeze(encoded_story_context_masked[-1,], dim=0)))

            gen_sentence.append(next_word_id)

            if next_word_id == full_stop_token:
                break

        return gen_sentence

    def encode_sentence_node(self, gen_sentence,  position, level, embedded_text_tensor, embedded_text_mask, indexed_tokens, parent=None):

        ctx_size = self.max_generated_sentence_length

        gen_sentence_length = len(gen_sentence)

        if not gen_sentence_length < self.min_sentence_length:

            if gen_sentence_length < ctx_size:
                gen_sentence_padded = gen_sentence + ([0] * (ctx_size - len(gen_sentence)))
            else:
                gen_sentence_padded = gen_sentence

                # Add newly generated context to sentences.
            gen_sentence_padded = torch.tensor(gen_sentence_padded).long()
            gen_sentence_padded = torch.unsqueeze(gen_sentence_padded, dim=0)
            generated_sentence_tensor = self.embedder(gen_sentence_padded.to(self._device))

            encoded_story, encoded_sentence, embedded_text_tensor, embedded_text_mask = self.encode_story(
                embedded_text_mask, embedded_text_tensor, generated_sentence_tensor,
                torch.squeeze(gen_sentence_padded, dim=0), position + level)
            sentence_length = len(gen_sentence)

            indexed_tokens[position + level] = torch.tensor(gen_sentence_padded).long().to(self._device)

            created_node = AnyNode(gold=False,
                                   story_tensor=encoded_story.cpu().detach(),
                                   sentence_tensor=encoded_sentence.cpu().detach(),
                                   embedded_text_tensor=embedded_text_tensor.cpu().detach(),
                                   embedded_text_mask=embedded_text_mask.cpu().detach(),
                                   indexed_tokens=indexed_tokens.cpu().detach(),
                                   token_ids=gen_sentence,
                                   level=level,
                                   sentence_text=[self.indexer.decoder[t].replace("</w>", "") for t in
                                                  gen_sentence if t in self.indexer.decoder],
                                   sentence_length=sentence_length, type="generated",
                                   parent=parent)
            return created_node

        return None

    def encode_story(self, embedded_text_mask, embedded_text_tensor, generated_sentence_tensor,
                     generated_sentence_indexed, position):
        # Pad up to the context length.

        new_embedded_text_tensor = torch.zeros(embedded_text_tensor.shape[0], max(embedded_text_tensor.shape[1],
                                                                                  generated_sentence_tensor.shape[1]),
                                               embedded_text_tensor.shape[2]).to(self._device)
        new_embedded_text_tensor[:, 0: embedded_text_tensor.shape[1], :] = embedded_text_tensor
        new_embedded_text_tensor[position, :] = 0
        new_embedded_text_tensor[position:, 0: generated_sentence_tensor.shape[1], :] = generated_sentence_tensor
        embedded_text_tensor = new_embedded_text_tensor

        new_embedded_text_mask = torch.zeros(embedded_text_tensor.shape[0], embedded_text_tensor.shape[1]).byte().to(
            self._device)
        new_embedded_text_mask[:, 0: embedded_text_mask.shape[1]] = embedded_text_mask
        new_embedded_text_mask[position] = 0

        new_masked_row = (generated_sentence_indexed > 0).byte()

        new_embedded_text_mask[position] = 0

        new_embedded_text_mask[position, 0: min(new_masked_row.shape[0], new_embedded_text_mask.shape[1])] = \
            new_masked_row[0: min(new_masked_row.shape[0], new_embedded_text_mask.shape[1])]

        embedded_text_mask = new_embedded_text_mask.long()
        # Encode as a story.

        encoded_story, encoded_sentence_vecs = self._model.encode_story_vectors(
            embedded_text_tensor.to(self._device), embedded_text_mask.to(self._device))

        encoded_story_at_position = encoded_story[position]
        encoded_sentence_at_position = torch.squeeze(encoded_sentence_vecs, dim=0)[position]

        encoded_story_at_position = torch.squeeze(encoded_story_at_position, dim=0)
        return encoded_story_at_position, encoded_sentence_at_position, embedded_text_tensor, embedded_text_mask

    def predict(self, embedded_text, story, indexed_length=None):

        logits = self._model._lm_model(embedded_text.to(self._device), story.to(self._device))
        logits =  torch.squeeze(logits, dim=0)

        if indexed_length is None:
            logits = logits[-1]
        else:
            logits = logits[indexed_length - 1]

        # Scale the logits by the temperature.
        next_word_id = random_sample(logits / self._generation_sampling_temperature, self.sample_top_k_words)

        return next_word_id

