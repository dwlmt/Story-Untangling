import torch
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.nn import Activation, util
from allennlp.nn.util import get_text_field_mask
from torch.nn import Module


class DynamicEntity(torch.nn.Module):
    # pylint: disable=line-too-long
    """ A derivative of EntityNLM.
    embedder : ``TextFieldEmbedder``, required.
        The embedder containing the embeddings to lookup.
    embedding_dim : ``int``
        Size of the embedding dimension.
    Returns
    -------
    Output contexts from the entity.
    """
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 embedding_dim: int,
                 context_dim: int,
                 hidden_dim: int = None,
                 entity_dim: int = None,
                 entity_encoder=None,
                 context_projection_layers: int = 1,
                 norm_type: int = 2.0,
                 dropout: float = 0.0,
                 activation: Activation = torch.nn.Sigmoid(),
                 ) -> None:
        super().__init__()
        self._embedder = embedder
        self._embedding_dim = embedding_dim

        self._entity_encoder = entity_encoder

        if hidden_dim:
            self._hidden_dim = hidden_dim
        else:
            self._hidden_dim = embedding_dim

        if entity_dim:
            self._entity_dim = entity_dim
        else:
            self._entity_dim = embedding_dim

        self._context_dim = context_dim
        self._norm_type = norm_type
        self._activation = activation

        self._context_transform = FeedForward(input_dim=self._context_dim, num_layers=context_projection_layers,
                                              hidden_dims=self._entity_dim, activations=activation, dropout=dropout)

        self._entity_delta_linear = torch.nn.Linear(self._entity_dim, self._entity_dim)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor,
                update_entities: bool = True):  # pylint: disable=arguments-differ
        embedded = self._embedder(inputs)

        context_transformed = self._context_transform(context)

        if update_entities:
            context_batch_size, context_feature_size = context_transformed.size()
            context_transformed = torch.unsqueeze(context_transformed, dim=1).expand(context_batch_size,
                                                                                     embedded.size()[1],
                                                                                     context_feature_size)

            entity_delta = torch.sigmoid(self._entity_delta_linear(embedded) * context_transformed)
            entities_updated = entity_delta * embedded + (1.0 - entity_delta) * context_transformed
            entities_updated_norm = torch.nn.functional.normalize(entities_updated, p=self._norm_type, dim=-1)

            for key, embedder in self._embedder._token_embedders.items():
                embedder.update(inputs[key], entities_updated_norm)
        else:
            entities_updated_norm = embedded

        if self._entity_encoder:
            mask = get_text_field_mask(inputs)
            entities_updated_norm = self._entity_encoder(entities_updated_norm, mask).to(entities_updated_norm.device)
        else:
            entities_updated_norm = torch.squeeze(entities_updated_norm, dim=1)

        return entities_updated_norm
