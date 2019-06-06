import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, get_mask_from_sequence_lengths
from allennlp.training.metrics import CategoricalAccuracy, Average
from torch import nn
from torch.nn import Dropout

from story_untangling.modules.gpt_lm import MixtureLM, FusionLM, BilinearLM

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("uncertain_reader")
class UncertainReader(Model):
    """
           Incremental reader than instead of predicting the next n sentences.

           Parameters
           ----------
           vocab : ``Vocabulary``, required
               Vocabulary containing source and target vocabularies. They may be under the same namespace
               (``tokens``) or the target tokens can have a different namespace, in which case it needs to
               be specified as ``target_namespace``.
           text_field_embedder : ``TextFieldEmbedder``, required
               Embeds a text field into a vector representation. Can be used to swap in or out Glove, ELMO, or BERT.
           story_seq2seq_encoder : ``Seq2SeqEncoder``,
               seq2Seq encode the story sentences into a higher level hierarchical abstraction.
           sentence_seq2seq_encoder : ``Seq2SeqEncoder``, optional (default = ``None``)
               seq2Seq encoder on the embedded feature of each sentence. Optional second level encoder on top of ELMO or language model.
           sentence_seq2vec_encoder : ``Seq2SeqEncoder``, optional (default = ``None``)
               seq2SeqVec encoder to use instead of the seq 2 seq encoder.
           fusion_seq2seq_encoder : ``Seq2SeqEncoder``, optional (default = ``None``)
               For concatenating and fusing the story context features to help the language model decode.
           target_seq2seq_encoder : ``Feedforward``,
               Target encoded used to fuse the vector. effectively a bridge to the future state.
           distance_weights: ``Tuple[float]``, optional (default = ``[1.0, 0.5, 0.25, 0.25]``)
                The numbers represent the weights to apply to n+1, n+2, n+3 is the loss function. The length how many sentence to look ahead in predictions.
           disc_length_regularizer : ``bool``, (optional, default=True)
                Regularizer that encourages the source and target vectors to be the same length.
           disc_length_regularizer_weight : ``float``, (optional, default=1.0)
                If the regularizer is set then the length to apply.
           dropout : ``float``, optional (default = ``None``)
                Dropout percentage to use.
           accuracy_top_k: ``Tuple[int]``, optional (default = ``[1, 3, 5, 10]``)
                For discriminatory loss calculate the the top k accuracy metrics.
           gen_loss: ``Tuple[int]``, optional (default = ``True``)
                Flag for the generative sequence to sequence loss.
           disc_loss: ``Tuple[int]``, optional (default = ``True``)
                Flag for the discriminatory loss.
           initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
               Used to initialize the model parameters.
           regularizer : ``RegularizerApplicator``, optional (default=``None``)
               If provided, will be used to calculate the regularization penalty during training.
           """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder = None,
                 story_seq2seq_encoder: Seq2SeqEncoder = None,
                 sentence_seq2seq_encoder: Seq2SeqEncoder = None,
                 sentence_seq2vec_encoder: Seq2VecEncoder = None,
                 fusion_seq2seq_encoder: Seq2SeqEncoder = None,
                 bilinear_fusion: bool = False,
                 target_feedforward: FeedForward = None,
                 story_feedforward: FeedForward = None,
                 dropout: float = None,
                 distance_weights: Tuple[float] = (1.0, 0.5, 0.25, 0.125),
                 disc_length_regularizer: bool = False,
                 disc_regularizer_weight: float = 0.1,
                 accuracy_top_k: Tuple[int] = (1, 10),
                 gen_loss: bool = True,
                 disc_loss: bool = True,
                 primary_token_namespace="openai_transformer",
                 lm_finetune_top_layers: int = 0,
                 gen_loss_weight: float = 1.0,
                 disc_loss_weight: float = 1.0,
                 full_output_scores: bool = False,
                 full_output_embeddings: bool = False,
                 flip_loss = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:
        super().__init__(vocab, regularizer)
        self._metrics = {}

        self._vocab = vocab
        self._text_field_embedder = text_field_embedder

        self._primary_token_namespace = primary_token_namespace

        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder
        self._sentence_seq2vec_encoder = sentence_seq2vec_encoder

        self._story_seq2seq_encoder = story_seq2seq_encoder
        self._target_feedforward = target_feedforward
        self._story_feedforward = story_feedforward

        feature_dim = story_seq2seq_encoder.get_output_dim()
        if self._story_feedforward:
            feature_dim = self._story_feedforward.get_output_dim()

        transformer = self._text_field_embedder._token_embedders[self._primary_token_namespace]._transformer

        self._flip_loss = flip_loss

        # Finetune the top n layers.

        if lm_finetune_top_layers > 0:
            for i, h in enumerate(reversed(transformer.h), start=1):
                for param in h.parameters():
                    param.requires_grad = True
                print(f"Finetune LM layer -{i}")
                if i == lm_finetune_top_layers:
                    break


        if bilinear_fusion:
            self._lm_model = BilinearLM(
                transformer=transformer,
                feature_dim=feature_dim,
                metrics=self._metrics,
                accuracy_top_k=accuracy_top_k)

        elif fusion_seq2seq_encoder is None:

            self._lm_model = MixtureLM(
                transformer=transformer,
                feature_dim=feature_dim,
                metrics=self._metrics,
                accuracy_top_k=accuracy_top_k)
        else:
            self._lm_model = FusionLM(
                transformer=transformer,
                encoder=fusion_seq2seq_encoder,
                metrics=self._metrics,
                accuracy_top_k=accuracy_top_k)


        self._distance_weights = distance_weights
        self._disc_length_regularizer = disc_length_regularizer
        self._disc_length_regularizer_weight = disc_regularizer_weight

        self._gen_loss = gen_loss
        self._disc_loss = disc_loss

        self._gen_loss_weight = gen_loss_weight
        self._disc_loss_weight = disc_loss_weight

        if dropout:
            self._dropout = Dropout(dropout)
        else:
            self._dropout = None

        self._accuracy_top_k = accuracy_top_k

        self._initializer = initializer
        self._regularizer = regularizer

        self._log_softmax = nn.LogSoftmax(dim=1)
        self._nll_loss = nn.NLLLoss(ignore_index=0)

        self._cosine_similarity = nn.CosineSimilarity()
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)

        self._full_output_scores = full_output_scores
        self.full_output_embedding = full_output_embeddings

        self.run_feedforwards = True

        self._metrics["accuracy_combined"] = Average()

        if self._disc_loss:

            for i in range(1, len(distance_weights) + 1):
                for top_n in self._accuracy_top_k:
                    self._metrics[f"disc_accuracy_{i}_{top_n}"] = CategoricalAccuracy(top_k=top_n)

                # self._metrics[f"entropy_{i}"] = Entropy()

                self._metrics[f"disc_correct_dot_product_avg_{i}"] = Average()
                self._metrics[f"disc_correct_log_prob_avg_{i}"] = Average()
                self._metrics[f"disc_correct_prob_avg_{i}"] = Average()
                self._metrics[f"disc_correct_similarity_cosine_avg_{i}"] = Average()
                self._metrics[f"disc_correct_distance_l1_avg_{i}"] = Average()
                self._metrics[f"disc_correct_distance_l2_avg_{i}"] = Average()

        if self._gen_loss:

            for top_n in self._accuracy_top_k:
                self._metrics[f"gen_accuracy_{top_n}"] = CategoricalAccuracy(top_k=top_n)

        initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor], metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
           Parameters
           ----------
           text : ``Dict[str, torch.LongTensor]``, required.
               The output of a ``TextField`` representing the text of
               the document.
           metadata : ``List[Dict[str, Any]]``, optional (default = None).
               A metadata dictionary for each instance in the batch.
           Returns
           -------
           An output dictionary consisting of:
           loss : ``torch.FloatTensor``, optional
               A scalar loss to be optimised.
           """
        output_dict = {}
        output_dict["metadata"] = metadata

        try:

            torch.set_printoptions(profile="full")

            # Because the batch has sentences need to reshape so can use the masking util function.
            text_mod = {}
            def_device = None
            batch_size = None
            for k, v in text.items():
                if k == self._primary_token_namespace:
                    batch_size, num_sentences, sentence_length = v.shape
                def_device = v.device
                text_mod[k] = v.view(batch_size * num_sentences, -1).to(self._lm_model.transformer.decoder.weight.device)

            embedded_text_tensor = self._text_field_embedder(text_mod).to(def_device)
            embedded_text_tensor = embedded_text_tensor.view(batch_size, num_sentences, embedded_text_tensor.shape[1], -1)

            masks_tensor =  text_mod["mask"]

            masks_tensor = masks_tensor.to(def_device).view(batch_size, num_sentences, -1)


            batch_encoded_stories = []
            batch_encoded_sentences = []

            for story_embedded_text, story_mask in zip(embedded_text_tensor.split(1), masks_tensor.split(1)):
                story_embedded_text = torch.squeeze(story_embedded_text, dim=0)
                story_mask = torch.squeeze(story_mask, dim=0)

                encoded_story = self.encode_story_vectors(story_embedded_text, story_mask)

                batch_encoded_stories.append(encoded_story)
                batch_encoded_sentences.append(story_embedded_text)

            batch_encoded_stories = torch.stack(batch_encoded_stories)

            loss = torch.tensor(0.0).to(batch_encoded_stories.device)

            source_encoded_stories = batch_encoded_stories
            target_encoded_stories = batch_encoded_stories

            if self._story_feedforward and self.run_feedforwards:
                source_encoded_stories = self._story_feedforward(source_encoded_stories)

            if self._target_feedforward and self.run_feedforwards:
                target_encoded_stories = self._target_feedforward(target_encoded_stories)

            flip = random.choice([True, False])

            if self.training:

                if self._disc_loss  or (self._flip_loss and flip):
                    disc_loss, disc_output_dict = self.calculate_discriminatory_loss(source_encoded_stories,
                                                                                     target_encoded_stories)
                    output_dict = {**output_dict, **disc_output_dict}
                    loss += (disc_loss * self._disc_loss_weight)

                if self._gen_loss or (self._flip_loss and not flip):
                    gen_loss, output_dict = self.calculate_gen_loss(source_encoded_stories, embedded_text_tensor, masks_tensor,
                                                                    text_mod, batch_size, num_sentences)
                    loss += (gen_loss * self._gen_loss_weight)

            if self.full_output_embedding:
                padded_embedded_text_tensor = torch.zeros((embedded_text_tensor.shape[0],embedded_text_tensor.shape[1],
                                                           masks_tensor.shape[-1], embedded_text_tensor.shape[-1])).float()
                padded_embedded_text_tensor[:, : embedded_text_tensor.shape[1], :]

                output_dict["source_encoded_stories"] = source_encoded_stories
                output_dict["embedded_text_tensor"] = padded_embedded_text_tensor
                output_dict["masks"] = masks_tensor

            output_dict["loss"] = loss
            return output_dict

        except Exception as e:
            print(text,metadata)
            raise e

    def encode_story_vectors(self, story_embedded_text, story_mask):

        # Create a mask that only has stories with sentences that go to the end of the batch.
        story_sentence_mask = torch.sum(story_mask, 1)
        story_sentence_mask = story_sentence_mask > 0
        story_sentence_mask = story_sentence_mask.to(story_embedded_text.device)
        # Get the final context  form each encoding.

        if self._sentence_seq2vec_encoder:
            encoded_sentence_vecs = self._sentence_seq2vec_encoder(story_embedded_text, story_mask)

        elif self._sentence_seq2seq_encoder:
            encoded_sentences = self._sentence_seq2seq_encoder(story_embedded_text, story_mask)
            story_mask[:, 0] = 1
            encoded_sentence_vecs = get_final_encoder_states(encoded_sentences,story_mask)

        encoded_sentence_vecs = torch.unsqueeze(encoded_sentence_vecs, dim=0)
        # Unflatten so can be stacked across stories
        story_sentence_mask = story_sentence_mask.unsqueeze(dim=0)
        encoded_story = self._story_seq2seq_encoder(encoded_sentence_vecs, story_sentence_mask)
        encoded_story = encoded_story.squeeze(dim=0)
        return encoded_story

    def calculate_gen_loss(self, encoded_stories, encoded_sentences, masks_tensor, target_text, batch_size,
                           num_sentences):

        output_dict = {}
        loss = torch.tensor(0.0).to(encoded_stories.device)

        encoded_sentences = encoded_sentences.view(encoded_sentences.shape[0] * encoded_sentences.shape[1],
                                                   encoded_sentences.shape[2], -1)

        masks_tensor = masks_tensor.view(masks_tensor.shape[0] * masks_tensor.shape[1], -1)
        encoded_stories = encoded_stories.view(
            encoded_stories.shape[0] * encoded_stories.shape[1], -1)

        batch_group_mask = self.batch_group_mask(batch_size, num_sentences)
        batch_group_mask = batch_group_mask.to(encoded_stories.device)

        batch_group_mask_target = batch_group_mask.clone()
        batch_group_mask_target[0] = 0
        batch_group_mask_target[-1] = 1

        batch_group_mask_story = batch_group_mask.clone()
        batch_group_mask_story[-1] = 0

        masks_tensor = masks_tensor[batch_group_mask_story].to(encoded_stories.device)
        encoded_stories = encoded_stories[batch_group_mask_story]

        encoded_sentences_target = encoded_sentences[batch_group_mask_target, :]

        non_empty_sentences = (masks_tensor.sum(dim=1) > 0)

        for k, v in target_text.items():
            if v is not None:
                v = v[batch_group_mask_target]
                v = v[non_empty_sentences]
                target_text[k] = v

        encoded_stories = encoded_stories[non_empty_sentences]
        encoded_sentences_target = encoded_sentences_target[non_empty_sentences]
        target_labels = target_text[self._primary_token_namespace]

        target_labels = target_labels[:, : encoded_sentences_target.shape[1]]

        loss += self._lm_model(encoded_sentences_target,
                               encoded_stories,
                               lm_labels=target_labels)

        return loss, output_dict

    def calculate_discriminatory_loss(self, encoded_stories, target_encoded_stories,
                                      sentence_loss=False):
        output_dict = {}
        loss = torch.tensor(0.0).to(encoded_stories.device)

        batch_size, sentence_num, feature_size = encoded_stories.shape

        target_encoded_stories = target_encoded_stories.to(encoded_stories.device)

        encoded_stories_flat = encoded_stories.view(batch_size * sentence_num, feature_size)
        target_encoded_sentences_flat = target_encoded_stories.view(batch_size * sentence_num, feature_size)

        dot_product_scores = self.calculate_logits(encoded_stories_flat, target_encoded_sentences_flat)

        dot_product_mask = (
                    1.0 - torch.diag(torch.ones(dot_product_scores.shape[0]), 0).float().to(dot_product_scores.device))
        dot_product_scores *= dot_product_mask

        if self._disc_length_regularizer:
            story_norm = encoded_stories_flat.norm(dim=1)
            target_norm_aligned = target_encoded_sentences_flat.norm(dim=1)

        distance_weights = self._distance_weights
        if sentence_loss:
            distance_weights = [1.0]
        for i, (distance_weights) in enumerate(distance_weights, start=1):

            # Use a copy to mask out elements that shouldn't be used.
            # This section excludes other correct answers for other distance ranges from the dot product.
            dot_product_scores_copy = dot_product_scores.clone()

            offsets = list(range(1, len(self._distance_weights) + 1))
            offsets = [o for o in offsets if o != i]
            for o in offsets:
                exclude_mask = (1 - torch.diag(torch.ones(dot_product_scores.shape[0]), o).float().to(
                    dot_product_scores.device))
                exclude_mask = exclude_mask[0:dot_product_scores.shape[0], 0:dot_product_scores.shape[1]]

                dot_product_scores_copy = dot_product_scores_copy * exclude_mask

            target_mask = torch.diag(torch.ones((batch_size * sentence_num) - i), i).byte().to(dot_product_scores.device)
            target_classes = torch.argmax(target_mask, dim=1).long().to(dot_product_scores.device)

            # Remove rows which spill over batches.
            batch_group_mask = self.batch_group_mask(batch_size, sentence_num, i=i)
            batch_group_mask = batch_group_mask.to(dot_product_scores.device)

            dot_product_scores_copy = dot_product_scores_copy[batch_group_mask,]
            target_mask = target_mask[batch_group_mask,]
            target_classes = target_classes[batch_group_mask]

            scores_softmax = self._log_softmax(dot_product_scores_copy)

            # Mask out sentences that are not present in the target classes.
            nll_loss = self._nll_loss(scores_softmax, target_classes)

            loss += nll_loss * distance_weights # Add the loss and scale it.

            # Regularizer to try and keep the vectors as similar lengths.
            if self._disc_length_regularizer:
                target_norm_aligned = target_norm_aligned[i:, ]
                story_norm_aligned = story_norm[story_norm.shape[0] - i]

                loss += (((
                                  story_norm_aligned - target_norm_aligned) ** 2 * self._disc_length_regularizer_weight).sum()
                         / batch_size) * distance_weights

            with torch.no_grad():

                if not self.training:

                    target_encoded_sentences_correct = target_encoded_sentences_flat[
                                                       i:, ]
                    batch_encoded_stories_correct = encoded_stories_flat[:encoded_stories_flat.shape[0] - i, :]

                    for top_k in self._accuracy_top_k:
                        self._metrics[f"disc_accuracy_{i}_{top_k}"](scores_softmax, target_classes)
                        self._metrics["accuracy_combined"](self._metrics[f"disc_accuracy_{i}_{top_k}"].get_metric())

                    self.similarity_metrics(batch_encoded_stories_correct, target_encoded_sentences_correct, i,
                                            output_dict)

                    # Some extra work just for metrics.
                    correct_scores = torch.masked_select(dot_product_scores_copy, target_mask)
                    correct_log_probs = torch.masked_select(scores_softmax, target_mask)
                    correct_probs = torch.exp(correct_log_probs)

                    output_dict[f"disc_correct_dot_product_{i}"] = correct_scores
                    output_dict[f"disc_correct_log_probs_{i}"] = correct_log_probs
                    output_dict[f"disc_correct_probs_{i}"] = correct_probs

                    self._metrics[f"disc_correct_dot_product_avg_{i}"](correct_scores.mean().item())
                    self._metrics[f"disc_correct_prob_avg_{i}"](correct_probs.mean().item())
                    self._metrics[f"disc_correct_log_prob_avg_{i}"](correct_log_probs.mean().item())

                if self._full_output_scores:
                    output_dict[f"disc_dot_products_{i}"] = dot_product_scores
                    output_dict[f"disc_log_probs_{i}"] = scores_softmax

        return loss, output_dict

    def calculate_logits(self, encoded_stories_flat, target_encoded_sentences_flat):
        dot_product_scores = torch.matmul(encoded_stories_flat,
                                          torch.t(target_encoded_sentences_flat))
        return dot_product_scores

    def batch_group_mask(self, batch_size, sentence_num, i=1):
        """ Mask out the last row in each batch as will not have a prediction for for the next row.
        """
        batch_group = torch.ones(sentence_num)
        batch_group.index_fill_(0, torch.tensor(list(range(sentence_num - i, sentence_num))), 0)
        batch_group = batch_group.unsqueeze(dim=0)
        batch_group = batch_group.expand(batch_size, sentence_num)
        batch_group = batch_group.contiguous().view(batch_size * sentence_num).byte()

        return batch_group

    def similarity_metrics(self, encoded_source, encoded_target, i, output_dict):
        # If using cosine similarity then these will be calculated on the unnormalised vectors. Since the measure don't make sense on the
        # normalised ones.
        with torch.no_grad():
            sim = self._cosine_similarity(encoded_source, encoded_target)
            output_dict[f"disc_correct_similarity_cosine_{i}"] = sim
            self._metrics[f"disc_correct_similarity_cosine_avg_{i}"](sim.mean().item())

            dist_l1 = self._l1_distance(encoded_source, encoded_target)
            output_dict[f"disc_correct_distance_l1_{i}"] = dist_l1
            self._metrics[f"disc_correct_distance_l1_avg_{i}"](dist_l1.mean().item())

            dist_l2 = self._l2_distance(encoded_source, encoded_target)
            output_dict[f"disc_correct_distance_l2_{i}"] = dist_l2
            self._metrics[f"disc_correct_distance_l2_avg_{i}"](dist_l2.mean().item())

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {metric_name: metric.get_metric(reset) for metric_name, metric in self._metrics.items()}

        return metrics
