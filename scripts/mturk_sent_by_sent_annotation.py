import argparse
import datetime

import boto3
import pandas

parser = argparse.ArgumentParser(
    description='Create tasks on MTurk using the API.')
parser.add_argument('--annotations-file', required=True, type=str, help="Link to the annotations file and codes.")
parser.add_argument('--output-file', required=True, type=str, help="The output file with the HIT ids data.")
parser.add_argument('--task-url', required=True, type=str, help="Link to the online annotations app.")
parser.add_argument('--max-assignments', required=False, type=int, default=100, help="Max assignments to create.")
parser.add_argument('--annotations-per-hit', required=False, type=int, default=5,
                    help="How many annotations per assignment.")
parser.add_argument('--reward', required=False, type=float, default=0.55, help="The reward per HIT.")
parser.add_argument('--access-key-id', required=True, help="AWS Access Key.")
parser.add_argument('--secret-access-key', required=True, type=str, help="AWS Access Key.")
parser.add_argument('--mturk-url', required=False, type=str,
                    default="https://mturk-requester-sandbox.us-east-1.amazonaws.com/", help="Mechanical turk link.")
parser.add_argument('--existing-counts', required=False, type=str, help="The existing count of story annotations complete. Subtract this from tasks to create.")

args = parser.parse_args()


def create_assignments(args):
    print(f"Create assignments: {args}")

    max_assignments = args["max_assignments"]
    ids_and_codes_df = pandas.read_csv(args["annotations_file"])

    worker_requirements = [
        # Restriction on too many annotations.
        {
            'QualificationTypeId': '31I5K8H9C3I6FRRHW1GPDMT1YCESP8',
            'Comparator': 'DoesNotExist',
            'ActionsGuarded': 'DiscoverPreviewAndAccept',
        },
        # Master worker qualification.
        {
            'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
            'Comparator': 'Exists',
            'ActionsGuarded': 'DiscoverPreviewAndAccept',
        },
        # Approval is >= 98%
        {
            'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [98],
            'ActionsGuarded': 'DiscoverPreviewAndAccept',
        },
        # The user has completed at least 1000 HITs/
        {
            'QualificationTypeId': '00000000000000000040',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [1000],
            'ActionsGuarded': 'DiscoverPreviewAndAccept',
        },
        # The user is an adult.
        {
            'QualificationTypeId': '00000000000000000060',
            'Comparator': 'EqualTo',
            'IntegerValues': [1],
            'ActionsGuarded': 'DiscoverPreviewAndAccept',
        }
    ]

    mturk = boto3.client('mturk',
                         aws_access_key_id=args["access_key_id"],
                         aws_secret_access_key=args["secret_access_key"],
                         region_name='us-east-1',
                         endpoint_url=args["mturk_url"],
                         )
    print(f"I have $ ${mturk.get_account_balance()['AvailableBalance']} in MTurk Account")

    hit_ids_data = []

    if args["existing_counts"] is not None:
        existing_counts_df = pandas.read_csv(args["existing_counts"])
    else:
        existing_counts_df = None

    total_storys_to_create = 0
    for i, row in ids_and_codes_df.iterrows():
        if i < max_assignments:
            hit_url = f"{args['task_url']}/?mturkCode={row['story_id']}-{row['code']}"
            print(f"Assignment to create: {row['story_id']}, {row['code']}, {hit_url}")

            external_question_xml = f'<ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd"><ExternalURL>{hit_url}</ExternalURL><FrameHeight>0</FrameHeight></ExternalQuestion>'
            print(f"Question XML: {external_question_xml}")

            annotations_per_hit = args["annotations_per_hit"]

            if existing_counts_df is not None:
                story_row = existing_counts_df.loc[existing_counts_df["story_id"] == row['story_id']]
                if len(story_row) > 0:
                    annotations_complete = story_row["num_of_workers"]
                    print(annotations_complete)
                    annotations_per_hit -= int(annotations_complete)

            print(row['story_id'], annotations_per_hit)

            total_storys_to_create += annotations_per_hit

            if annotations_per_hit > 0:

                new_hit = mturk.create_hit(
                    Title=f'Story dramatic tension reading sentence by sentence {row["story_id"]}',
                    Description='Read a short story and record the level of dramatic tension per sentence. Will take 5-10 minutes per hit. Fluent English speakers required. Some violent, sexual or other disturbing content may be present in the stories.',
                    Keywords='story, narrative, storytelling, annotation, research, nlp, reading',
                    Reward=f'{args["reward"]}',
                    MaxAssignments=annotations_per_hit,
                    LifetimeInSeconds=1209600,  # Two weeks
                    AssignmentDurationInSeconds=3600,  # One hour
                    AutoApprovalDelayInSeconds=604800,  # 5 Days
                    Question=external_question_xml,
                    QualificationRequirements=worker_requirements
                )

            hit_ids_data.append({"hit_id": new_hit['HIT']['HITId'], "story_id": row['story_id'], "code": row['code']})


    print(total_storys_to_create)

    hit_df = pandas.DataFrame(data=hit_ids_data)
    hit_df.to_csv(args["output_file"])


create_assignments(vars(args))
