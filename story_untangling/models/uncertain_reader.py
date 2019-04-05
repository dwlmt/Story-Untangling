import logging

import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models import Model, SimpleSeq2Seq
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from typing import Dict, Any, List, Optional

from allennlp.nn.util import get_text_field_mask

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
           sentence_seq2seq_encoder : ``Seq2SeqEncoder``,
               seq2Seq encoder on the embedded feature of each sentence.
           story_seq2seq_encoder : ``Seq2SeqEncoder``,
               seq2Seq encode the story sentences into a higher level hierarchical abstraction.
           dropout : ``float``, optional (default = ``None``)
                Dropout percentage to use.
           initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
               Used to initialize the model parameters.
           regularizer : ``RegularizerApplicator``, optional (default=``None``)
               If provided, will be used to calculate the regularization penalty during training.
           """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_seq2seq_encoder: Seq2SeqEncoder,
                 story_seq2seq_encoder: Seq2SeqEncoder,
                 dropout: float = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:
        super().__init__(vocab)
        self._vocab = vocab
        self._text_field_embedder =  text_field_embedder
        self._sentence_seq2seq_encoder = sentence_seq2seq_encoder
        self._story_seq2seq_encoder = story_seq2seq_encoder

        self._dropout = dropout
        self._initializer = initializer
        self._regularizer = regularizer

        initializer(self)


    def forward(self,
                text: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
           Parameters
           ----------
           text : ``Dict[str, torch.LongTensor]``, required.
               The output of a ``TextField`` representing the text of
               the document.
           metadata : ``List[Dict[str, Any]]``, optional (default = None).
               A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
               from this dictionary, which respectively have the original text and the annotated gold coreference
               clusters for that instance.
           Returns
           -------
           An output dictionary consisting of:
           loss : ``torch.FloatTensor``, optional
               A scalar loss to be optimised.
           """

        output_dict = {}
        output_dict["metadata"] = metadata

        embedded_text_tensor = self._text_field_embedder(text)
        batch_size, num_sentences, sentence_length, embedded_feature_size = embedded_text_tensor.shape

        # Because the batch has sentences need to reshape so can use the masking util function.
        text_mod = {}
        for k, v in text.items():
            text_mod[k] = v.view(batch_size * num_sentences, sentence_length, -1)
        masks_tensor = get_text_field_mask(text_mod)
        masks_tensor = masks_tensor.view(batch_size, num_sentences, sentence_length, -1)

        # TODO: Parallelize using multiple GPUS if available rather than using a loop.

        for story_embedded_text, story_mask in zip(embedded_text_tensor.split(1),masks_tensor.split(1)):
            story_embedded_text = torch.squeeze(story_embedded_text, dim=0)
            story_mask = torch.squeeze(story_mask, dim=0)

            story_encoded_sentences = self._sentence_seq2seq_encoder(story_embedded_text, story_mask)
            print(story_encoded_sentences.shape)


        loss = torch.tensor(0.5) # TODO: A dummy to return loss until the dataset is working for training.
        output_dict["loss"] = loss

        return output_dict

