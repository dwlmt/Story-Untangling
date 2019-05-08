import itertools

import torch
from allennlp.common import JsonDict, tqdm
from allennlp.common.util import sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.nn.util import get_final_encoder_states
from allennlp.predictors import Predictor
from anytree import AnyNode, RenderTree

full_stop_token = 239


def random_sample(logits: torch.Tensor) -> int:
    d = torch.distributions.Categorical(logits=logits)
    sampled = d.sample()

    return sampled.item()


def choose_max(logits: torch.Tensor, ) -> int:
    p, i = torch.max(logits, dim=-1)
    return i.item()



@Predictor.register("uncertain_reader_gen_predictor")
class UncertainReaderGenPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.coreference_resolution.ReadingThoughtsPredictor(` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)

        self._model.full_output_embedding = True

        self.embedder = model._text_field_embedder._token_embedders["openai_transformer"]
        self.embedder._top_layer_only = True

        self.tokenizer = dataset_reader._tokenizer
        self.indexer =  dataset_reader._token_indexers["openai_transformer"]

        self.sentences_to_unroll = 2
        self.samples_per_level = 5


    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        embedded_text_tensor = outputs["embedded_text_tensor"]
        embedded_text_tensor = torch.from_numpy(embedded_text_tensor).cpu()
        embedded_text_mask = torch.from_numpy(outputs["masks"]).cpu()

        for position, (text_field, encoded_story) in enumerate(zip(instance["text"],
                                                                         torch.from_numpy(outputs["source_encoded_stories"]).cpu())):

            root = AnyNode(id="root")

            text_tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())

            text_to_gen_from = text_tensor_dict['openai_transformer']
            ctx_size = text_to_gen_from.size(0)

            sentence_length = torch.sum(embedded_text_mask[position]).item()

            text_to_gen_from = text_to_gen_from[0:sentence_length]

            if sentence_length == 0:
                continue

            for sample_num in range(self.samples_per_level):
                out =  self.generate_sentence(position, embedded_text_tensor, embedded_text_mask,  encoded_story, text_to_gen_from,
                                              sentence_length, ctx_size, recursion=self.sentences_to_unroll - 1)
                print(position, out)

        return {}#sanitize(outputs)

    def generate_sentence(self, position, embedded_text_tensor, embedded_text_mask, encoded_story, text_to_gen_from, sentence_length, ctx_size, recursion=0):
        gen_sentence = []

        print(embedded_text_tensor.shape)
        encoded_text_merged = embedded_text_tensor[position, :, :]
        print(encoded_text_merged.shape)

        for i in range(ctx_size - 1):

            if len(gen_sentence) > 0:

                text_tensor_merged = torch.cat((text_to_gen_from, torch.tensor(gen_sentence).long()))

                sentence_length = text_tensor_merged.shape[-1]

                if text_tensor_merged.shape[0] > ctx_size:
                    text_tensor_merged = (text_tensor_merged[1:ctx_size + 1])
                else :
                    text_tensor_merged = torch.cat(
                        (text_tensor_merged, torch.tensor([0] * (ctx_size - len(text_tensor_merged))).long()))

                encoded_text_merged = self.embedder(torch.unsqueeze(text_tensor_merged, dim=0).long())

            next_word_id = self.predict(encoded_text_merged, encoded_story, sentence_length, ctx_size)

            if not next_word_id:
                break

            gen_sentence.append(next_word_id)

            if next_word_id == full_stop_token:
                print("End of sentence")
                break

        decoded_sentence = [self.indexer.decoder[t] for t in gen_sentence if t in self.indexer.decoder]

        # Pad up to the context length.
        gen_sentence_length = len(gen_sentence)
        if gen_sentence_length < ctx_size:
            gen_sentence_padded = gen_sentence + ([0] * (ctx_size - len(gen_sentence)))
        else:
            gen_sentence_padded = gen_sentence

        gen_sentence_padded = torch.tensor(gen_sentence_padded).long()
        gen_sentence_padded = torch.unsqueeze(gen_sentence_padded, dim=0)

        generated_sentence_tensor = self.embedder(gen_sentence_padded)
        print("Generated Sentence Tensor", generated_sentence_tensor.shape)

        # New sentence can be longer than the existing batch so need to resize the representation.
        #if generated_sentence_tensor.shape[1] > embedded_text_tensor.shape[1]:
        new_embedded_text_tensor = torch.zeros(embedded_text_tensor.shape[0], max(embedded_text_tensor.shape[1],
                                                                                  generated_sentence_tensor.shape[1]),
                                               embedded_text_tensor.shape[2])
        new_embedded_text_tensor[:, 0 : embedded_text_tensor.shape[1], :] = embedded_text_tensor
        new_embedded_text_tensor[position:, 0: generated_sentence_tensor.shape[1], :] = generated_sentence_tensor
        embedded_text_tensor = new_embedded_text_tensor

        new_embedded_text_mask = torch.zeros(embedded_text_tensor.shape[0], embedded_text_tensor.shape[1])
        new_embedded_text_mask[: , 0: embedded_text_mask.shape[1]] = embedded_text_mask
        embedded_text_mask = new_embedded_text_mask.long()

        #else:
            # Copy as don't want to change the original state.
        #    embedded_text_tensor = embedded_text_tensor.clone()
        #    embedded_text_tensor[position] = generated_sentence_tensor

        embedded_text_tensor[position] = generated_sentence_tensor
        print(embedded_text_tensor.shape, embedded_text_mask.shape)
        encoded_sentences, encoded_story, story_sentence_mask = self._model.encode_story_vectors(embedded_text_tensor, embedded_text_mask)
        print(encoded_sentences.shape, encoded_story.shape, story_sentence_mask.shape)

        encoded_story = encoded_story[position]

        if self._model._story_feedforward:
            encoded_story = self._model._story_feedforward(encoded_story)

        encoded_story = torch.squeeze(encoded_story, dim=0)

        generated_sentence_tensor = torch.squeeze(generated_sentence_tensor, dim=0)
        encoded_story = torch.squeeze(encoded_story)

        if recursion > 0:
            for i in range(self.samples_per_level):

                next_position = position + 1

                # Don't project past the end of the story.
                if next_position == embedded_text_tensor.shape[0]:
                    break

                out = self.generate_sentence(next_position, embedded_text_tensor, embedded_text_mask, encoded_story,
                                             torch.tensor(gen_sentence).long(), len(gen_sentence), ctx_size, recursion=recursion-1)
                print(out)


        return generated_sentence_tensor, encoded_story, gen_sentence, decoded_sentence

    def predict(self, embedded_text, story, sentence_length, ctx_size):
        story = torch.unsqueeze(story, dim=0)
        logits = self._model._lm_model(embedded_text, story)
        logits =  torch.squeeze(logits, dim=0)
        logits = logits[min(sentence_length - 1, ctx_size - 1), :]
        next_word_id = random_sample(logits)

        return next_word_id

