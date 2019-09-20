''' Match annotations data with the database.

'''
import argparse

import dataset
import pandas
import tqdm
from nltk.metrics.distance import edit_distance

engine_kwargs = {"pool_recycle": 3600, "connect_args": {'timeout': 300, "check_same_thread": False}}


def match_annotations(args):
    print(args)

    pandas_list = []
    for filename in args["source_csvs"]:
        df = pandas.read_csv(filename, index_col=None, header=0)
        pandas_list.append(df)

    concat_df = pandas.concat(pandas_list, ignore_index=True, sort=False, axis=0)

    print(concat_df.columns)
    # print(tabulate(concat_df, headers='keys', tablefmt='psql'))

    text_to_id_dict = {}
    id_to_story = {}

    text = concat_df["Input.text"]
    answer_story_ids = concat_df["Answer.storyId"]

    original_database = args["original_database"]
    dataset_db = f"sqlite:///{original_database}"

    annotation_mapping_records = {}
    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

        for orig_id, t in zip(answer_story_ids, text):

            try:
                orig_id = int(orig_id)
                text_formatted = t.replace("<newline>", "")
                text_to_id_dict[text_formatted] = orig_id

            except ValueError:
                pass

    print(text_to_id_dict)

    database = args["database"]
    dataset_db = f"sqlite:///{database}"

    with dataset.connect(dataset_db, engine_kwargs=engine_kwargs) as db:

        for k, v in text_to_id_dict.items():

            best_total = int(1e6)
            best_match = None

            for story in tqdm.tqdm(list(db["story"]), desc="Exact match loop"):
                sentences = [s for s in db.query(
                    f"SELECT * from sentence where story_id = {story['id']} and sentence_num < {args['num_sentence']} ORDER By sentence_num")]
                # print(sentence["text"])

                block = ' '.join([s["text"] for s in sentences])
                compare = k[0:len(block)]

                if block == compare:
                    best_total = 0
                    best_match = v

            if best_total != 0:
                for story in tqdm.tqdm(list(db["story"]), desc="Fuzzy match loop"):
                    sentences = [s for s in db.query(
                        f"SELECT * from sentence where story_id = {story['id']} and sentence_num < {args['num_sentence']} ORDER By sentence_num")]
                    # print(sentence["text"])

                    block = ' '.join([s["text"] for s in sentences])
                    compare = k[0:len(block)]

                    if block == compare:
                        total = 0
                    else:
                        total = edit_distance(block, compare)

                    if v == story['id']:
                        print(f"Same Story: {v} - {story['id']}, {total}")

                    if total < best_total:
                        print(f"New best match: {v} - {story['id']}, {total}")
                        best_total = total
                        best_match = v

            if best_match:

                if best_match not in annotation_mapping_records:
                    # print(f"Matched Story Text on {sentence}")
                    annotation_mapping_records[best_match] = {"story_id": best_match,
                                                              "annotation_story_id": v}
                    print(annotation_mapping_records[best_match], best_total)

        db["story_annotation_map"].insert_many(annotation_mapping_records.values())
        db["story_annotation_map"].create_index(['story_id'])
        print(annotation_mapping_records.values())


parser = argparse.ArgumentParser(
    description='Add per sentence sentiment information to the database.')
parser.add_argument('--source-csvs', required=True, nargs="+", type=str, help="The RAW Amazon Mechanical Turk ")
parser.add_argument('--original-database', required=True, type=str, help="The database to connect to.")
parser.add_argument('--database', required=True, type=str, help="The database to connect to.")
parser.add_argument('--num-sentence', default=5, type=int, help="Number of sentence: 5")

args = parser.parse_args()

match_annotations(vars(args))
