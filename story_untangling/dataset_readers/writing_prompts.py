import asyncio
import logging
from typing import Dict, List, Union, Any

import dataset
import numpy
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

from story_untangling.dataset_readers.dataset_features import create_dataset_db, negative_sentence_sampler
from story_untangling.dataset_readers.dataset_utils import dual_window

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("writing_prompts")
class WritingPromptsDatasetReader(DatasetReader):
    """
    This is derived from the Sequence to Sequence AllenNLP base class. Unlike normal sequence to sequence the source and
    target are sliding windows of context text and text to predict.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    add_start_end_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    sentence_context_window : int, (optional, default=2)
        Reads as a sliding windows over sentences. How many sentences to use as a context for each sentence to predict.
    sentence_predictive_window : int, (optional, default=1)
        Reads as a sliding windows over sentences. How many sentences to predict.
    target_negative : bool, (optional, default=True)
        Whether to yield a negative target context or not.
    dataset_path : str, (optional, default=./dataset-cache/)
        Path for caching the dataset features. Will reuse rather than rebuilding if it finds the SQLLite dataset.
    use_existing_cached_db : bool, (optional, default=True)
        Reused an existing cached database db if it exists or delete and recreate.
    db_discriminator : str (optional, default=def)
        Allow multiple databases to be kept separate rather than wiping over each other.
    save_sentiment: bool (optional, default=True)
        Whether to save sentence level sentiment when creating the dataset.
    ner_model : str (optional, default=None)
        AllenNLP NER model to run for features.
    coreference_model : str (optional, default=None)
        AllenNLP Coreference model.
    min_story_sentences : int, (optional, default=0)
        Min number of sentences a story must have to be included.
    max_story_sentences : int, (optional, default=1000000)
        Max number of sentences a story must have to be included.
    positional_features : bool, (optional, default=True)
        Should encode positional features in the source.
    truncate_sequence_length : int, (optional, default=250)
        Target sequences longer than this value will be truncated to this value (cutting starting from the end).
        0 indicates length is unlimited. Value must be greater than or equal to 0.
    cuda_device : List[Int] (optional, default=-1)
        List of CUDA devices. This is needed in cases such as NER and coreferencing where preprocessing benefits from CUDA.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_end_token: bool = True,
                 sentence_context_window: int = 2,
                 sentence_predictive_window: int = 1,
                 target_negative: bool = True,
                 dataset_path: str = "./dataset-cache/",
                 use_existing_cached_db: bool = True,
                 db_discriminator="def",
                 save_sentiment: bool = True,
                 ner_model: str = None,
                 coreference_model: str = None,
                 min_story_sentences: int = 0,
                 max_story_sentences: int = 10 * 6,
                 positional_features: bool = True,
                 truncate_sequence_length: int = 250,
                 cuda_device: Union[List[int], int] = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._add_start_end_token = add_start_end_token
        self._sentence_context_window = sentence_context_window
        self._sentence_predictive_window = sentence_predictive_window
        self._target_negative: bool = target_negative
        self._dataset_path = dataset_path
        self._use_existing_cached_db = use_existing_cached_db
        self._db_discriminator = db_discriminator
        self._save_sentiment = save_sentiment
        self._ner_model = ner_model
        self._coreference_model = coreference_model
        self._min_story_sentences = min_story_sentences
        self._max_story_sentences = max_story_sentences
        self._positional_features = positional_features
        self._truncate_sequence_length = truncate_sequence_length
        self._truncate_sequences = (truncate_sequence_length != 0)
        self._cuda_device = cuda_device

    @overrides
    def _read(self, file_path):

        loop = asyncio.get_event_loop()
        dataset_db = loop.run_until_complete(
            create_dataset_db(dataset_path=self._dataset_path, db_discriminator=self._db_discriminator,
                              should_save_sentiment=self._save_sentiment,
                              file_path=file_path, use_existing_database=self._use_existing_cached_db,
                              ner_model=self._ner_model,
                              coreference_model=self._coreference_model,
                              cuda_device=self._cuda_device))

        db = dataset.connect(dataset_db, engine_kwargs={"pool_recycle": 3600})

        negative_sampler = negative_sentence_sampler(db)

        stories = db.query(
            f'SELECT * FROM story  WHERE sentence_num >= {self._min_story_sentences} '
            f'AND sentence_num <= {self._max_story_sentences} ORDER BY id')
        for story in stories:
            story_id = story["id"]
            # Id will be the same as the sentence num as they are inserted as a batch in sequence.
            sentences = [s for s in db.query(f'SELECT * FROM sentence WHERE story_id = {story_id} ORDER BY id')]
            if len(sentences) == 0:
                logging.warning(f"Story has no sentences: {story_id}")
                continue



            sentence_nums = [s["sentence_num"] for s in sentences]
            for source_indices, target_indices, absolute_position, relative_position in dual_window(sentence_nums,
                                                                                                context_size=self._sentence_context_window,
                                                                                                predictive_size=self._sentence_predictive_window,
                                                                                                num_of_sentences=story[
                                                                                                    "sentence_num"]):

                source_sequence = [sentences[i] for i in source_indices if i is not None]
                target_sequence = [sentences[i] for i in target_indices if i is not None]

                if self._target_negative:
                    negative_sequence = []
                    for i in range(self._sentence_predictive_window):
                        negative_sequence.append(next(negative_sampler))


                metadata = {"story_id": story_id, "absolute_position": absolute_position,
                            "relative_position": relative_position, "sentence_number": story["sentence_num"]}


                yield self.text_to_instance(source_sequence, target_sequence, negative_sequence,
                                            metadata=metadata)

    @overrides
    def text_to_instance(self,
                         source_sequence: Dict[str,Any],
                         target_sequence: Dict[str, Any],
                         negative_sequence: Dict[str, Any],
                         metadata: Dict[str, Any] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        field_dict = {}

        def tokenize(tokens, tokenizer, indexer, ner):
            tokenized_text = tokenizer(tokens)

            # If separate ner tags are provided then replace.
            if ner and len(ner) > 0:

                # If provided then swap in the database saved NER tag.
                for t, n in zip(tokenized_text, ner.split()):
                    ner_tag = n.split('-')[-1].strip()
                    t.ent_type_ = ner_tag

            if self._add_start_end_token:
                tokenized_text.insert(0, Token(START_SYMBOL))
                tokenized_text.append(Token(END_SYMBOL))

            if len(tokenized_text) > self._truncate_sequence_length and self._truncate_sequences:
                tokenized_text = tokenized_text[:self._truncate_sequence_length]

            token_field = TextField(tokenized_text, indexer)

            if len(tokenized_text) == 0:
                token_field = token_field.empty_field()

            return token_field

        source_tokens = " ".join([s["text"] for s in source_sequence])
        source_ner = " ".join([s["ner_tags"] for s in source_sequence if "ner_tags" in s])

        target_tokens = " ".join([s["text"] for s in target_sequence])
        target_ner = " ".join([s["ner_tags"] for s in target_sequence if "ner_tags" in s])


        field_dict['source_tokens'] = tokenize(source_tokens, self._source_tokenizer.tokenize,
                                               self._source_token_indexers, source_ner)

        if target_tokens is not None:
            field_dict['target_tokens'] = tokenize(target_tokens, self._target_tokenizer.tokenize,
                                                   self._target_token_indexers, target_ner)


        if self._target_negative:
            negative_tokens = " ".join([s["text"] for s in negative_sequence])
            negative_ner = " ".join([s["ner_tags"] for s in target_sequence if "ner_tags" in s])

            field_dict['negative_tokens'] = tokenize(negative_tokens, self._target_tokenizer.tokenize,
                                                     self._target_token_indexers, negative_ner)

        # Wrap in an array there isn't a single value scalar field.
        source_features = []
        target_features = []
        negative_features = []

        # It only makes sense to include in the source features as otherwise it gives away the correct answers.
        if self._positional_features:
            source_features.append(metadata["absolute_position"])
            source_features.append(metadata["relative_position"])


        if self._save_sentiment:
            source_features.extend(self.construct_global_sentiment_features(
                source_sequence))
            target_features.extend(self.construct_global_sentiment_features(
                target_sequence))

            negative_features.extend(self.construct_global_sentiment_features(
                negative_sequence))

        if len(source_features) > 0:
            field_dict["source_features"] = ArrayField(numpy.array(source_features))

        if len(target_features) > 0:
            field_dict["target_features"] = ArrayField(numpy.array(target_features))

        if len(negative_features) > 0:
            field_dict["negative_features"] = ArrayField(numpy.array(negative_features))


        field_dict["metadata"] = MetadataField(metadata)

        return Instance(field_dict)

    def construct_global_sentiment_features(self, source_sequence):
        vader_sentiment = 0.0
        textblob_polarity = 0.0
        textblob_subjectivity = 0.0
        for s in source_sequence:
            if "vader_sentiment" not in s or "textblob_polarity" not in s or "textblob_subjectivity" not in s:
                continue

            vader_sentiment += s["vader_sentiment"]
            textblob_polarity += s["textblob_polarity"]
            textblob_subjectivity += s["textblob_subjectivity"]
        vader_sentiment /= len(source_sequence)
        textblob_polarity /= len(source_sequence)
        textblob_subjectivity /= len(source_sequence)
        return textblob_polarity, textblob_subjectivity, vader_sentiment
