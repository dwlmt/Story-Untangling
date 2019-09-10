import argparse
import json
import random
import string

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--database', required=True, type=str, help="The testset database")
parser.add_argument('--annotation-story-ids', required=True, help='The file with the ids of the annotation stories.')
parser.add_argument('--output-json', required=True, type=str, help="The output JSON file location.")


args = parser.parse_args()

def generate_random_code(length=12):
    """Generate a random string of letters and digits """
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for i in range(length))


def extract_annotations(args):
    print(f"Extract annotations to JSON: {args}")

    output_dict = {}
    stories_list = []

    import pandas as pd
    data = pd.read_csv(args["annotation_story_ids"])
    story_ids = data["story_id"].unique().tolist()

    metadata_list = []

    import dataset
    with dataset.connect(f'sqlite:///{args["database"]}', engine_kwargs={"pool_recycle": 3600}) as db:

        for story_id in story_ids:
            story_dict = {}
            print(f"{story_id}")
            story_dict["story_id"] = int(story_id)
            code = str(generate_random_code())
            story_dict["code"] = code

            metadata_list.append({"story_id": story_id, "code": code})

            sentences = [s for s in db.query(
                f'SELECT * FROM sentence INNER JOIN sentence_lang on sentence.id = sentence_lang.sentence_id '
                f'WHERE sentence.story_id = {story_id} and sentence_lang.lang = "en" '
                f'and sentence_lang.nonsense = false and sentence_lang.ascii_chars=true ORDER BY id')]

            story_dict["sentences"] = sentences
            stories_list.append(story_dict)

        output_dict["stories"] = stories_list

        with open(args["output_json"], 'w') as outfile:
            json.dump(output_dict, outfile)

        metadata_df = pd.DataFrame(data=metadata_list)
        metadata_df.to_csv(args["output_json"].replace(".json","_meta.csv"))

extract_annotations(vars(args))