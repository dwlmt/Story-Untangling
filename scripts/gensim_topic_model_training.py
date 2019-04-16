import argparse

import gensim
import spacy
from gensim import corpora
from gensim.models import HdpModel


def train_topics(args):
    print(f"Arguments: {args}")

    nlp = spacy.load("en", disable=["parser", "ner"])

    files = args["text"]
    lines = extract_stories(files)

    def tozenize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        allowed_postags = set(allowed_postags)
        docs = nlp.pipe(texts)
        text_tokens = []
        for doc in docs:
            tokens = [token.lemma_ for token in doc if
                      token.pos_ in allowed_postags and not token.is_punct and not token.is_stop]
            text_tokens.append(tokens)
        return text_tokens

    docs = tozenize(lines, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print("Preprocessed Docs")

    bigram = gensim.models.Phrases(docs, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[docs], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    docs = make_bigrams(docs)
    docs = make_trigrams(docs)

    print("Create Dictionary")
    # Create Dictionary
    corpus_dict = corpora.Dictionary(docs)
    # Create Corpus
    texts = docs
    # Term Document Frequency
    corpus = [corpus_dict.doc2bow(text) for text in texts]

    print("Train Model")
    hdp = HdpModel(corpus, corpus_dict)

    print(hdp.print_topics(num_topics=50, num_words=20))

    hdp.save(args["target"])


def extract_stories(files):
    lines = []

    for text_file in files:

        with open(text_file) as f:
            for line in f:
                line = line.replace("<newline>", " ")
                lines.append(line)

    return lines


parser = argparse.ArgumentParser(
    description='Training a HDP Topic Model.')
parser.add_argument('--text', nargs='+', type=str,
                    help="List of text files to train the topic model. The assumed format is one new line per file as per WRITING PROMPTS")
parser.add_argument('--target', required=True, type=str, help="Output the saved weights of the Topic Model")

args = parser.parse_args()
train_topics(vars(args))
