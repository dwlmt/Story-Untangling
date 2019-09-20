import argparse

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from jsonlines import jsonlines

parser = argparse.ArgumentParser(
    description='Download results from Mechanical Turk.')
parser.add_argument('--firebase-key-path', required=True, type=str, help="The path to the JSON key.")
parser.add_argument('--output-file', required=True, type=str,
                    help="The output file to save the data to in JSON lines format.")
parser.add_argument('--collection-name', required=True, type=str,
                    help="The name of the Firebase collection to download.")


def download_firebase_collection(args):
    print(f"Download Firebase collection: {args}")

    collection_data = []

    cred = credentials.Certificate(args["firebase_key_path"])
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    collection_ref = db.collection(args['collection_name'])
    docs = collection_ref.stream()

    for doc in docs:
        doc_dict = {}
        doc_dict["id"] = doc.id
        doc_dict["document"] = doc.to_dict()
        doc_dict["collection"] = args['collection_name']
        collection_data.append(doc_dict)

    with jsonlines.open(args['output_file'], mode='w') as writer:
        for d in collection_data:
            writer.write(d)


args = parser.parse_args()

download_firebase_collection(vars(args))
