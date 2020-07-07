import argparse
import xml.dom.minidom
from urllib.parse import urlparse

import pandas


parser = argparse.ArgumentParser(
    description='Convert broken annotations from Mechanical Turk')
parser.add_argument('--assignments-file', required=True, type=str, help="Link to the annotations file and codes.")
parser.add_argument('--output-file', required=True, type=str, help="The output file with the HIT ids data.")

args = parser.parse_args()


def fix_assignments(args):

    story_ids = []
    codes = []

    assignment_df = pandas.read_json(args["assignments_file"])
    for i, row in assignment_df.iterrows():
        root = xml.dom.minidom.parseString(row["Question"])

        print(root)

        for node in root.getElementsByTagName('ExternalURL'):
            url = node.firstChild.wholeText
            url_parsed = urlparse(url)
            query = url_parsed.query
            query = query.replace('mturkCode=','')
            story_id, code = query.split("-")
            print(story_id, code)
            story_ids.append(story_id)
            codes.append(code)

    assignment_df["story_id"] = story_ids
    assignment_df["code"] = codes
    assignment_df["hit_id"] = assignment_df["HITId"]

    assignment_df.to_csv(args["output_file"])


fix_assignments(vars(args))