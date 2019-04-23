import itertools

import torch
import tqdm
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel

from torch import nn, Tensor
from allennlp.modules.openai_transformer import OpenaiTransformer
from allennlp.modules import FeedForward
from torch.nn import CrossEntropyLoss
from typing import Dict, Any, List

from story_untangling.modules.utils import random_sample, LRUCache


class FusionLM(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, transformer: OpenaiTransformer, feature: FeedForward = None,
                 feature_input_dim=None, metrics: Dict[str, Any] = None,
                 accuracy_top_k: List = None):
        super(FusionLM, self).__init__()
        self.transformer = transformer
        self.feature_input_dim = None
        self.feature = feature
        self._metrics = metrics
        self._accuracy_top_k = accuracy_top_k

        if self.feature:
            self.feature = feature
            # Match the output vocab on the linear layer.
            feature_input_dim = feature.get_output_dim()
        self.feature_decoder = torch.nn.Linear(in_features=feature_input_dim,
                                               out_features=self.transformer.decoder.out_features, bias=False)

        self.decoder = self.transformer.decoder

        self._log_softmax = nn.LogSoftmax(dim=1)
        self.loss = CrossEntropyLoss(ignore_index=-1)

    def forward(self, hidden_states, feature_contexts=None, lm_labels=None):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)

        decoded = self.decoder(hidden_states)

        if self.feature is not None and feature_contexts is not None:
            feature_contexts = self.feature(feature_contexts)

        if feature_contexts is not None:
            feature_contexts = self.feature_decoder(feature_contexts)
            if lm_labels is not None:
                if len(decoded.shape) == 3:
                    feature_contexts = feature_contexts.unsqueeze(dim=1)

                decoded *= feature_contexts

        if lm_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = decoded[..., :-1, :].contiguous().cpu()
            if feature_contexts is not None:
                shift_logits *= feature_contexts.cpu()
            shift_labels = lm_labels[..., 1:].contiguous().cpu()

            source_logits = shift_logits.view(-1, shift_logits.size(-1))
            target_classes = shift_labels.view(-1)

            loss = self.loss(source_logits, target_classes)

            if self._metrics:
                scores_softmax = self._log_softmax(source_logits)
                with torch.no_grad():
                    for top_k in self._accuracy_top_k:
                        self._metrics[f"gen_accuracy_{top_k}"](scores_softmax, target_classes)

            return loss.to(hidden_states.device)
        return decoded
