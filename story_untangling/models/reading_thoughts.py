import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Average

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
       initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
       """

    def __init__(self,
                 vocab: Vocabulary,

                 source_embedder: TextFieldEmbedder,
                 source_encoder: Seq2VecEncoder,
                 target_embedder: TextFieldEmbedder = None,
                 target_encoder: Seq2VecEncoder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ) -> None:
        super().__init__(vocab)
        self._vocab = vocab
        self._source_embedder = source_embedder
        self._source_encoder = source_encoder
        self._target_embedder = target_embedder or source_embedder
        self._target_encoder = target_encoder or source_encoder

        self._log_softmax = nn.LogSoftmax(dim=1)
        self._nll_loss = nn.NLLLoss()

        self.metrics = {
            "neighbour_accuracy": CategoricalAccuracy(),
            "neighbour_accuracy5": CategoricalAccuracy(top_k=5),
            "neighbour_correct_score_avg": Average(),
            "neighbour_correct_log_prob_avg": Average(),
            "neighbour_correct_prob_avg": Average(),
            "random_accuracy": CategoricalAccuracy(),
            "random_accuracy5": CategoricalAccuracy(top_k=5),
            "random_correct_score_avg": Average(),
            "random_correct_log_prob_avg": Average(),
            "random_correct_prob_avg": Average()
        }

        initializer(self)

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                negative_tokens: Dict[str, torch.LongTensor] = None,
                absolute_position: Dict[str, Any] = None,
                relative_position: Dict[str, Any] = None,
                metadata: Dict[str, Any] = None,
                epoch: Dict[str, Any] = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the target sequences.
        Parameters
        ----------
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

        """
        output_dict = {}

        embedded_source = self._source_embedder(source_tokens)

        batch_size, _, _ = embedded_source.size()

        source_mask = get_text_field_mask(source_tokens)
        encoded_source = self._source_encoder(embedded_source, source_mask)

        embedded_target = self._target_embedder(target_tokens)
        target_mask = get_text_field_mask(target_tokens)
        encoded_target = self._target_encoder(embedded_target, target_mask)

        scores = torch.matmul(encoded_source, torch.t(encoded_target))
        loss = self._calculate_loss(batch_size, scores, output_dict)

        if negative_tokens:
            embedded_negative = self._target_embedder(negative_tokens)
            negative_mask = get_text_field_mask(negative_tokens)
            encoded_negative = self._target_encoder(embedded_negative, negative_mask)

            # This is replacing one of the negative random samples with the correct output.
            # It is hacky but computationally efficient meaning one of the random samples isn't used.
            neg_scores = torch.matmul(encoded_source, torch.t(encoded_negative))
            identity = torch.eye(batch_size, dtype=torch.long).to(scores.device)
            neg_identity = (identity != 1)
            comb_scores = torch.zeros(scores.shape).to(scores.device)
            comb_scores += scores * identity.float()
            comb_scores += neg_scores * neg_identity.float()

            neg_loss = self._calculate_loss(batch_size, comb_scores, output_dict, metrics_prefix="negative")
            loss += neg_loss

        output_dict["loss"] = loss


        return output_dict

    def _calculate_loss(self, batch_size, scores, output_dict, metrics_prefix="neighbour"):

        loss = torch.tensor(0.0).to(scores.device)

        output_dict["{metrics_prefix}_scores"] = scores

        # The correct answer should correspond to the same position in the batch.
        target_labels = torch.zeros(batch_size, batch_size, dtype=torch.long).to(scores.device)
        identity = torch.eye(batch_size, dtype=torch.long).to(scores.device)
        target_labels += identity
        correct_mask = target_labels == 1
        # Calculate the loss
        scores_softmax = self._log_softmax(scores)
        for t in target_labels:
            loss += self._nll_loss(scores_softmax, t)

            self.metrics[f"{metrics_prefix}_accuracy"](scores_softmax, t)
            self.metrics[f"{metrics_prefix}_accuracy5"](scores_softmax, t)

        # Some extra work just for metrics.
        output_dict[f"{metrics_prefix}_scores_softmax"] = scores_softmax
        output_dict[f"{metrics_prefix}_correct_score"] = scores[correct_mask]
        target_correct_probability = scores_softmax[correct_mask]
        output_dict[f"{metrics_prefix}_correct_probability"] = target_correct_probability.mean().item()
        self.metrics[f"{metrics_prefix}_correct_score_avg"](
            output_dict[f"{metrics_prefix}_correct_score"].mean().item())
        probs = torch.exp(target_correct_probability)
        self.metrics[f"{metrics_prefix}_correct_log_prob_avg"](target_correct_probability.mean().item())
        self.metrics[f"{metrics_prefix}_correct_prob_avg"](probs.mean().item())

        return loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
