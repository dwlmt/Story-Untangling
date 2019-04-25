import torch

from torch import nn
from allennlp.modules.openai_transformer import OpenaiTransformer
from allennlp.modules import FeedForward
from torch.nn import NLLLoss, CrossEntropyLoss
from typing import Dict, Any, List


class MixtureLM(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, transformer: OpenaiTransformer, feature_dim: int, metrics: Dict[str, Any] = None,
                 accuracy_top_k: List = None):
        super(MixtureLM, self).__init__()
        self.transformer = transformer
        self._metrics = metrics
        self._accuracy_top_k = accuracy_top_k

        # Trainable weight for combining the language model.
        self._lm_weighting = torch.tensor([0.5], requires_grad=True)

        self._feature_decoder = torch.nn.Linear(in_features=feature_dim,
                                                out_features=self.transformer.decoder.out_features, bias=False)

        self._decoder = self.transformer.decoder

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = NLLLoss(ignore_index=-1)

    def forward(self, lm_hidden_states, feature_hidden_states, lm_labels=None):

        lm_weighting = self._lm_weighting.clamp(min=0.001, max=0.999).to(lm_hidden_states.device)

        feature_logits = self._feature_decoder(feature_hidden_states)

        if len(lm_hidden_states.shape) == 3:
            feature_logits = feature_logits.unsqueeze(dim=1)

        lm_logits = self._decoder(lm_hidden_states)
        lm_logits = (lm_logits * lm_weighting) + (feature_logits * (1.0 - lm_weighting))

        if lm_labels is not None:

            # Shift so that tokens < n predict n
            lm_logits = lm_logits[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()

            # Flatten the tokens and classes
            lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            lm_labels = lm_labels.view(-1)

            scores_softmax = self.log_softmax(lm_logits)
            loss = self.loss(scores_softmax, lm_labels)

            if self._metrics and not self.training:
                with torch.no_grad():
                    for top_k in self._accuracy_top_k:
                        self._metrics[f"gen_accuracy_{top_k}"](scores_softmax, lm_labels)

            return loss.to(lm_hidden_states.device)

        return lm_logits
