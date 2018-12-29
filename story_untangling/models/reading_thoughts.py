import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, SimilarityFunction
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.common.checks import check_dimensions_match

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
                 similarity_function: SimilarityFunction = None,
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

        self._similarity_function = similarity_function

        # TODO: Rework to allow other similarity based functions to be used.
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._nll_loss = nn.NLLLoss()

        self.metrics = {
            "neighbour_accuracy": CategoricalAccuracy(),
            "neighbour_accuracy5": CategoricalAccuracy(top_k=5),
            "neighbour_accuracy3": CategoricalAccuracy(top_k=3),
            "neighbour_correct_score_avg": Average(),
            "neighbour_correct_log_prob_avg": Average(),
            "neighbour_correct_prob_avg": Average(),
            "neighbour_correct_similarity_avg": Average(),
            "negative_accuracy": CategoricalAccuracy(),
            "negative_accuracy3": CategoricalAccuracy(top_k=3),
            "negative_accuracy5": CategoricalAccuracy(top_k=5),
            "negative_correct_score_avg": Average(),
            "negative_correct_log_prob_avg": Average(),
            "negative_correct_prob_avg": Average(),
            "negative_correct_similarity_avg": Average(),
        }

        initializer(self)

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor],
                negative_tokens: Optional[Dict[str, torch.LongTensor]] = None,
                source_features: Optional[Dict[str, Any]] = None,
                target_features: Optional[Dict[str, Any]] = None,
                negative_features: Optional[Dict[str, Any]] = None,
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
        source_features: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        target_features: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        negative_features: ``Opt[Dict[str, Any]]``, optional
            Global source features that are applied across the whole context.
        metadata: ``Opt[Dict[str, Any]]``, optional
            metadata with story information.
        epoch: ``Opt[Dict[str, Any]]``, optional
            the epoch of the run.

        """

        # TODO: Refactor in separate module when more stable.
        def reading_encoder(features=None, tokens=None, embedder=None, encoder=None, feedforward=None):
            embedded_source = embedder(tokens)
            size, _, _ = embedded_source.size()
            source_mask = get_text_field_mask(tokens)
            encoded = encoder(embedded_source, source_mask)
            if feedforward:
                if features is not None:
                    encoded = torch.cat((encoded, features), dim=-1)
                encoded = feedforward(encoded)
            return encoded, size

        output_dict = {}

        output_dict["metadata"] = metadata

        encoded_source, batch_size = reading_encoder(source_features, source_tokens,
                                                     self._source_embedder,
                                                     self._source_encoder,
                                                     self._source_feedforward)
        loss = torch.tensor(0.0).to(encoded_source.device)

        if target_tokens:
            encoded_target, _ = reading_encoder(target_features, target_tokens,
                                                self._target_embedder,
                                                self._target_encoder,
                                                self._target_feedforward)

            output_dict["encoded_source"] = encoded_source
            output_dict["encoded_target"] = encoded_target

            scores = torch.matmul(encoded_source, torch.t(encoded_target))
            loss += self._calculate_loss(batch_size, scores, output_dict)

            # If there is a custom similarity defined then output using this similarity.
            if self._similarity_function:
                with torch.no_grad():
                    sim = self._similarity_function(encoded_source, encoded_target)
                    output_dict[f"neighbour_correct_similarity"] = sim
                    self.metrics[f"neighbour_correct_similarity_avg"](sim.mean().item())

        if negative_tokens:
            encoded_negative, _ = reading_encoder(negative_features, negative_tokens,
                                                  self._target_embedder,
                                                  self._target_encoder,
                                                  self._target_feedforward)

            output_dict["encoded_negative"] = encoded_negative

            # This is replacing one of the negative random samples with the correct output.
            # It is hacky but computationally efficient meaning one of the random samples isn't used.
            neg_scores = torch.matmul(encoded_source, torch.t(encoded_negative))
            identity = torch.eye(batch_size, dtype=torch.long).to(scores.device)
            neg_identity = (identity != 1)
            comb_scores = torch.zeros(scores.shape).to(scores.device)
            comb_scores += scores * identity.float()
            comb_scores += neg_scores * neg_identity.float()

            loss += self._calculate_loss(batch_size, comb_scores, output_dict, metrics_prefix="negative")

            # If there is a custom similarity defined then output using this similarity.
            if self._similarity_function:
                with torch.no_grad():
                    sim = self._similarity_function(encoded_source, encoded_negative)
                    output_dict[f"negative_correct_similarity"] = sim
                    self.metrics[f"negative_correct_similarity_avg"](sim.mean().item())

        output_dict["loss"] = loss

        return output_dict

    def _calculate_loss(self, batch_size, scores, output_dict, metrics_prefix="neighbour"):

        # The correct answer should correspond to the same position in the batch.
        identity = torch.eye(batch_size, dtype=torch.long).to(scores.device)
        target_classes = torch.argmax(identity, dim=1)  # Get indices and NLLLoss needs these.
        # Calculate the loss
        scores_softmax = self._log_softmax(scores)

        loss = self._nll_loss(scores_softmax, target_classes)

        with torch.no_grad():
            # Some extra work just for metrics.
            correct_mask = (torch.zeros(batch_size, batch_size, dtype=torch.long).to(scores.device) + identity) == 1
            correct_scores = scores[correct_mask]
            correct_log_probs = scores_softmax[correct_mask]
            correct_probs = torch.exp(correct_log_probs)

            output_dict[f"{metrics_prefix}_scores"] = scores
            output_dict[f"{metrics_prefix}_scores_log_softmax"] = scores_softmax
            output_dict[f"{metrics_prefix}_correct_score"] = correct_scores
            output_dict[f"{metrics_prefix}_correct_log_probs"] = correct_log_probs
            output_dict[f"{metrics_prefix}_correct_probs"] = correct_probs

            self.metrics[f"{metrics_prefix}_accuracy"](scores_softmax, target_classes)
            self.metrics[f"{metrics_prefix}_accuracy3"](scores_softmax, target_classes)
            self.metrics[f"{metrics_prefix}_accuracy5"](scores_softmax, target_classes)
            self.metrics[f"{metrics_prefix}_correct_score_avg"](correct_scores.mean().item())
            self.metrics[f"{metrics_prefix}_correct_prob_avg"](correct_probs.mean().item())
            self.metrics[f"{metrics_prefix}_correct_log_prob_avg"](correct_log_probs.mean().item())

        return loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
