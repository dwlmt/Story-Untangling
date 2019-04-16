import argparse

import dataset
import spacy
from gensim.models import HdpModel

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}


def add_topics(args):
    print(args)

    nlp = spacy.load("en", disable=["parser", "ner"])

    def tozenize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        allowed_postags = set(allowed_postags)
        docs = nlp.pipe(texts)
        text_tokens = []
        for doc in docs:
            tokens = [token.lemma_ for token in doc if
                      token.pos_ in allowed_postags and not token.is_punct and not token.is_stop]
            text_tokens.append(tokens)
        return text_tokens

    model = HdpModel.load(args["topic_model"])
    corpus_dict = model.id2word

    topics = model.show_topics(num_topics=args["num_topics"], num_words=args["num_terms"], log=False, formatted=False)

    topics_to_save = []
    for topic in topics:
        topic_dict = {}
        topic_terms = ", ".join([t[0] for t in topic[1]])
        topic_dict["topic_id"] = int(topic[0])
        topic_dict["terms"] = topic_terms

        topics_to_save.append(topic_dict)

    database = args["database"]
    dataset_db = f"sqlite:///{database}"
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:
        db.create_table("corpus_topics")

        topic_ids = db["corpus_topics"].insert_many(topics_to_save)
        print(topic_ids)

        print(topics_to_save)

        batch = []
        for sentence in db['sentence']:
            batch.append(sentence)

            if len(batch) == args["batch_size"]:
                insert_corpus_sentence_links(batch, corpus_dict, db, model, tozenize)
                batch = []

        if len(batch) > 0:
            insert_corpus_sentence_links(batch, corpus_dict, db, model, tozenize)

        db["corpus_topics_sentences"].create_index(['sentence_id'])
        db["corpus_topics_sentences"].create_index(['topic_id'])


def insert_corpus_sentence_links(batch, corpus_dict, db, model, tozenize):
    print("Insert batch")
    texts = tozenize([s["text"] for s in batch])
    text_vectors = [corpus_dict.doc2bow(text) for text in texts]
    topic_sentences_to_insert = []
    for v, sentence in zip(text_vectors, batch):
        text_topics = model[v]

        for tt in text_topics:
            topic_sentence = {}
            topic_sentence["sentence_id"] = sentence["id"]
            topic_sentence["topic_id"] = int(tt[0]) + 1  # The topic is indexed from 1 in sqllite.
            topic_sentence["proportion"] = float(tt[1])
            topic_sentences_to_insert.append(topic_sentence)
    db["corpus_topics_sentences"].insert_many(topic_sentences_to_insert)


parser = argparse.ArgumentParser(
    description='Add per sentence sentiment information to the database.')
parser.add_argument('--database', required=True, type=str, help="Output the saved weights of the Topic Model")
parser.add_argument('--topic-model', required=True, type=str, help="Path to the HDP Gensim Topic Model")
parser.add_argument('--batch-size', type=int, default=100, help="Output the saved weights of the Topic Model")
parser.add_argument('--num-topics', type=int, default=50, help="Number of topics to use from HDP. Default: 50")
parser.add_argument('--num-terms', type=int, default=20, help="Top N number of terms to extract. Default: 20")

args = parser.parse_args()

add_topics(vars(args))
