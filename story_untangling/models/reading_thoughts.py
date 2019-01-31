import logging
from typing import Dict, Any, Optional, Callable

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, SimilarityFunction
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Average
from pandas import *

from story_untangling.modules.dynamic_entity import DynamicEntity

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("reading_thoughts")
class ReadingThoughts(Model):
    """
       Simple derivative of Quick-Thoughts to train and encoder based on predicting a following a sentence rather than surrounding contex.
       Also allows different contexts to be used to encode the source and target sequences.
       Parameters
       ----------
       vocab : ``Vocabulary``, required
           Vocabulary containing source and target vocabularies. They may be under the same namespace
           (``tokens``) or the target tokens can have a different namespace, in which case it needs to
           be specified as ``target_namespace``.
       source_embedder : ``TextFieldEmbedder``, required
           Embedder for source context sequences.
       source_encoder : ``Seq2VecEncoder``, required
           The encoder for the source context text.
       target_embedder : ``TextFieldEmbedder``, (optional, default=source_embedder)
           Embedder for the target sequences, defaults to the same as source if not specified.
       target_encoder : ``Seq2VecEncoder``, (optional, default=source_encoder)
           The encoder for the target sequence, if not specified used the target one.
       source_feedforward : ``FeedForward``
           The source feedfoward network for projection and merging the context for the source to merge global and
        sequence based features.
       target_feedforward : ``FeedForward``
        The target feedfoward network for projection and merging the context for the source to merge global and
        sequence based features.
       story_embedder : ``TextFieldEmbedder``, optional
           Embedder for the story as a whole.
       story_embedding_dim : ``int``, optional
           Embedder for the story as a whole.
       dot_product_loss : ``bool``, (optional, default=True)
           Use dot product when calculating the Quick Thoughts based loss.
       length_regularizer : ``bool``, (optional, default=True)
           Regularizer that encourages the source and target vectors to be the same length.
       length_regularizer_weight : ``float``, (optional, default=1.0)
           Weight for the regularizer in the loss function.
       cosine_loss : ``bool``, (optional, default=False)
           Use cosine when calculating the Quick Thoughts loss instead of plain dot product.
       full_output_score : ``bool``, (optional, default=False)
           Embedder for the story as a whole.
       full_output_embeddings : ``bool``, (optional, default=False)
           Output the embedding vectors in the output.
       initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
           Used to initialize the model parameters.
       """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 source_encoder: Seq2VecEncoder,
                 target_embedder: TextFieldEmbedder = None,
                 target_encoder: Seq2VecEncoder = None,
                 source_feedforward: FeedForward = None,
                 target_feedforward: FeedForward = None,
                 story_embedder: TextFieldEmbedder = None,
                 story_embedding_dim: int = None,
                 entity_context_dim: int = None,
                 entity_embedder: TextFieldEmbedder = None,
                 entity_embedding_dim: int = None,
                 entity_encoder: Seq2VecEncoder = None,
                 similarity_function: SimilarityFunction = None,
                 cosine_loss: bool = False,
                 length_regularizer: bool = True,
                 length_regularizer_weight: float = 1.0,
                 full_output_score: bool = False,
                 full_output_embeddings: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ) -> None:
        super().__init__(vocab)
        self._vocab = vocab
        self._source_embedder = source_embedder
        self._source_encoder = source_encoder
        self._source_feedforward = source_feedforward

        # Share the network with the source if not specified.
        self._target_embedder = target_embedder or source_embedder
        self._target_encoder = target_encoder or source_encoder
        self._target_feedforward = target_feedforward or source_feedforward

        if story_embedder:
            self._story_encoder = DynamicEntity(embedder=story_embedder, embedding_dim=story_embedding_dim,
                                                context_dim=entity_context_dim)
        else:
            self._story_encoder = None

        if entity_embedder:
            self._entity_encoder = DynamicEntity(embedder=entity_embedder, embedding_dim=entity_embedding_dim,
                                                 context_dim=entity_context_dim,
                                                 entity_encoder=entity_encoder)
        else:
            self._entity_encoder = None

        self._cosine_loss = cosine_loss

        self._length_regularizer = length_regularizer
        self._length_regularizer_weight = length_regularizer_weight

        self._cosine_similarity = nn.CosineSimilarity()
        self._l2_distance = nn.PairwiseDistance(p=2)
        self._l1_distance = nn.PairwiseDistance(p=1)
        self._similarity_function = similarity_function

        self._full_output_score = full_output_score
        self._full_output_embeddings = full_output_embeddings

        # TODO: Rework to allow other similarity based functions to be used.
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._nll_loss = nn.NLLLoss()

        self.metrics = {
            "neighbour_accuracy": CategoricalAccuracy(),
            "neighbour_accuracy5": CategoricalAccuracy(top_k=5),
            "neighbour_accuracy3": CategoricalAccuracy(top_k=3),
            "neighbour_correct_dot_product_avg": Average(),
            "neighbour_correct_log_prob_avg": Average(),
            "neighbour_correct_prob_avg": Average(),
            "neighbour_correct_similarity_cosine_avg": Average(),
            "neighbour_correct_distance_l1_avg": Average(),
            "neighbour_correct_distance_l2_avg": Average(),
            "negative_accuracy": CategoricalAccuracy(),
            "negative_accuracy3": CategoricalAccuracy(top_k=3),
            "negative_accuracy5": CategoricalAccuracy(top_k=5),
            "negative_correct_dot_product_avg": Average(),
            "negative_correct_log_prob_avg": Average(),
            "negative_correct_prob_avg": Average(),
            "negative_correct_similarity_cosine_avg": Average(),
            "negative_correct_distance_l1_avg": Average(),
            "negative_correct_distance_l2_avg": Average(),
        }

        initializer(self)

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor],
                negative_tokens: Optional[Dict[str, torch.LongTensor]] = None,
                story: Dict[str, torch.LongTensor] = None,
                source_features: Optional[Dict[str, Any]] = None,
                target_features: Optional[Dict[str, Any]] = None,
                negative_features: Optional[Dict[str, Any]] = None,
                source_coreferences: Dict[str, torch.LongTensor] = None,
                target_coreferences: Dict[str, torch.LongTensor] = None,
                negative_coreferences: Dict[str, torch.LongTensor] = None,
                metadata: Optional[Dict[str, Any]] = None,
                epoch: Optional[Dict[str, Any]] = None
                ) -> Dict[str, torch.Tensor]:
        """
        source_tokens: ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()`` applied on the source
            ``TextField``. This will be passed through a ``TextFieldEmbedder``
            and then through an encoder.
        target_tokens: ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()`` applied on the source
            ``TextField``. This will be passed through a ``TextFieldEmbedder``
            and then through an encoder. These are the correct target sentences and neighbouring sentences in the text.
        negative_tokens: ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()`` applied on the source
            ``TextField``. This will be passed through a ``TextFieldEmbedder``
            and then through an encoder. These are sampled or selected from non-negative text.
        story: ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()`` applied on the source
            ``TextField``. This will be passed through a ``TextFieldEmbedder``
            and then through an encoder.
        source_features: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        target_features: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        negative_features: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        source_coreferences: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        target_coreferences: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        negative_coreferences: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        metadata: ``Opt[Dict[str, Any]]``, optional
            metadata with story information.
        epoch: ``Opt[Dict[str, Any]]``, optional
            the epoch of the run.
        """

        # TODO: Refactor in separate module when more stable.
        def reading_encoder(features=None,
                            tokens=None,
                            story=None,
                            coreferences=None,
                            embedder=None,
                            encoder=None,
                            feedforward=None,
                            story_encoder=None, dynamic_entity_encoder=None, update_dynamic_entities=False,
                            ):

            embedded_source = embedder(tokens)
            batch_size, _, _ = embedded_source.size()
            source_mask = get_text_field_mask(tokens)
            encoded = encoder(embedded_source, source_mask)
            orig_encoded = encoded

            if features is not None:
                encoded = torch.cat((encoded, features), dim=-1)

            if story_encoder and story:
                story_encoded = story_encoder(story, orig_encoded, update_dynamic_entities)
                encoded = torch.cat((encoded, story_encoded), dim=-1)

            if dynamic_entity_encoder and coreferences:
                entity_encoded = dynamic_entity_encoder(coreferences, orig_encoded, update_dynamic_entities)
                encoded = torch.cat((encoded, entity_encoded), dim=-1)

            if feedforward:
                encoded = feedforward(encoded)

            return encoded, batch_size

        output_dict = {}

        output_dict["metadata"] = metadata

        encoded_source, batch_size = reading_encoder(features=source_features,
                                                     tokens=source_tokens,
                                                     story=story,
                                                     coreferences=source_coreferences,
                                                     embedder=self._source_embedder,
                                                     encoder=self._source_encoder,
                                                     feedforward=self._source_feedforward,
                                                     story_encoder=self._story_encoder,
                                                     dynamic_entity_encoder=self._entity_encoder,
                                                     update_dynamic_entities=True)

        if self._full_output_embeddings:
            output_dict["source_embeddings"] = encoded_source.tolist()

        loss = torch.tensor(0.0).to(encoded_source.device)

        if target_tokens:
            # With targets don't updated the dynamic entities but rely on those from the 
            encoded_target, _ = reading_encoder(features=target_features, tokens=target_tokens,
                                                embedder=self._target_embedder,
                                                encoder=self._target_encoder,
                                                coreferences=target_coreferences,
                                                story=story,
                                                feedforward=self._target_feedforward,
                                                story_encoder=self._story_encoder,
                                                dynamic_entity_encoder=self._entity_encoder,
                                                update_dynamic_entities=False
                                                )

            if self._full_output_embeddings:
                output_dict["target_embeddings"] = encoded_target.tolist()

            batch_loss, scores = self._calculate_loss(batch_size, encoded_source, encoded_target, output_dict,
                                         full_output_score=self._full_output_score)
            loss += batch_loss


            # If there is a custom similarity defined then output using this similarity.
            self.similarity_metrics(encoded_source, encoded_target, "neighbour", output_dict)

        if negative_tokens:
            encoded_negative, _ = reading_encoder(features=negative_features, tokens=negative_tokens,
                                                  embedder=self._target_embedder,
                                                  encoder=self._target_encoder,
                                                  coreferences=negative_coreferences,
                                                  story=story,
                                                  feedforward=self._target_feedforward,
                                                  story_encoder=self._story_encoder,
                                                  dynamic_entity_encoder=self._entity_encoder,
                                                  update_dynamic_entities=False
                                                  )

            if self._full_output_embeddings:
                output_dict["negative_embeddings"] = encoded_negative.tolist()



            batch_loss, scores = self._calculate_loss(batch_size, encoded_source, encoded_negative, output_dict,
                                                      metrics_prefix="negative", full_output_score=self._full_output_score, existing_scores=scores)
            loss += batch_loss

            self.similarity_metrics(encoded_source, encoded_target, "negative", output_dict)

        output_dict["loss"] = loss

        return output_dict

    def similarity_metrics(self, encoded_source, encoded_target, name, output_dict):
        # If using cosine similarity then these will be calculated on the unnormalised vectors. Since the measure don't make sense on the
        # normalised ones.
        with torch.no_grad():
            sim = self._cosine_similarity(encoded_source, encoded_target)
            output_dict[f"{name}_correct_similarity_cosine"] = sim
            self.metrics[f"{name}_correct_similarity_cosine_avg"](sim.mean().item())

            dist_l1 = self._l1_distance(encoded_source, encoded_target)
            output_dict[f"{name}_correct_distance_l1"] = dist_l1
            self.metrics[f"{name}_correct_distance_l1_avg"](dist_l1.mean().item())

            dist_l2 = self._l2_distance(encoded_source, encoded_target)
            output_dict[f"{name}_correct_distance_l2"] = dist_l2
            self.metrics[f"{name}_correct_distance_l2_avg"](dist_l2.mean().item())

    def _calculate_loss(self, batch_size, encoded_source, encoded_target, output_dict,
                        metrics_prefix="neighbour", full_output_score=False, existing_scores=None):

        # The dot product L2 normalised dot roduct is the equivalent of
        if self._cosine_loss:
            encoded_source = torch.nn.functional.normalize(encoded_source, dim=-1)
            encoded_target = torch.nn.functional.normalize(encoded_target, dim=-1)

        if not metrics_prefix == "negative":
            dot_product_scores = torch.matmul(encoded_source, torch.t(encoded_target))
        else:

            # This is replacing one of the negative random samples with the correct output.
            # It is hacky but computationally efficient meaning one of the random samples isn't used.
            neg_scores = torch.matmul(encoded_source, torch.t(encoded_target))
            identity = torch.eye(batch_size, dtype=torch.long).to(existing_scores.device)
            neg_identity = (identity != 1)
            comb_scores = torch.zeros(existing_scores.shape).to(existing_scores.device)
            comb_scores += existing_scores * identity.float()
            comb_scores += neg_scores * neg_identity.float()
            dot_product_scores = comb_scores


        # The correct answer should correspond to the same position in the batch.
        identity = torch.eye(batch_size, dtype=torch.long).to(dot_product_scores.device)
        target_classes = torch.argmax(identity, dim=1)  # Get indices and NLLLoss needs these.
        # Calculate the loss

        loss = torch.tensor(0.0).to(encoded_source.device)

        scores_softmax = self._log_softmax(dot_product_scores)
        loss += self._nll_loss(scores_softmax, target_classes)

        # The length regularizer penalizes cases where the correct source and targets aren't the same length.
        # Redundant when applying a cosine loss as the vectors will be normalized for length anyway.
        if self._length_regularizer and not self._cosine_loss:

            source_norm = encoded_source.norm(dim=-1)
            target_norm = encoded_target.norm(dim=-1)

            loss += ((source_norm - target_norm) ** 2 * self._length_regularizer_weight).sum() / batch_size

        with torch.no_grad():

            # Some extra work just for metrics.
            correct_mask = (torch.zeros(batch_size, batch_size, dtype=torch.long).to(
                dot_product_scores.device) + identity) == 1
            correct_scores = dot_product_scores[correct_mask]
            correct_log_probs = scores_softmax[correct_mask]
            correct_probs = torch.exp(correct_log_probs)

            output_dict[f"{metrics_prefix}_correct_dot_product"] = correct_scores
            output_dict[f"{metrics_prefix}_correct_log_probs"] = correct_log_probs
            output_dict[f"{metrics_prefix}_correct_probs"] = correct_probs

            if full_output_score:
                output_dict[f"{metrics_prefix}_dot_products"] = dot_product_scores
                output_dict[f"{metrics_prefix}_log_probs"] = scores_softmax

            self.metrics[f"{metrics_prefix}_accuracy"](scores_softmax, target_classes)
            self.metrics[f"{metrics_prefix}_accuracy3"](scores_softmax, target_classes)
            self.metrics[f"{metrics_prefix}_accuracy5"](scores_softmax, target_classes)
            self.metrics[f"{metrics_prefix}_correct_dot_product_avg"](correct_scores.mean().item())
            self.metrics[f"{metrics_prefix}_correct_prob_avg"](correct_probs.mean().item())
            self.metrics[f"{metrics_prefix}_correct_log_prob_avg"](correct_log_probs.mean().item())

        return loss, dot_product_scores

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
