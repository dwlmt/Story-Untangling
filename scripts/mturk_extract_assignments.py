import argparse

import boto3
import pandas
import xmltodict

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--assignments-file', required=True, type=str, help="Link to the annotations file and codes.")
parser.add_argument('--output-file', required=True, type=str, help="The output file with the HIT ids data.")
parser.add_argument('--access-key-id', required=False, type=str, default="AKIAI33ZD245LOFYMYAQ", help="AWS Access Key.")
parser.add_argument('--secret-access-key', required=False, type=str, default="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",
                    help="AWS Access Key.")
parser.add_argument('--mturk-url', required=False, type=str,
                    default="https://mturk-requester-sandbox.us-east-1.amazonaws.com/", help="Mechanical turk link.")

args = parser.parse_args()


def create_assignments(args):
    print(f"Create assignments: {args}")

    mturk = boto3.client('mturk',
                         aws_access_key_id=args["access_key_id"],
                         aws_secret_access_key=args["secret_access_key"],
                         region_name='us-east-1',
                         endpoint_url=args["mturk_url"],
                         )
    print(f"I have $ ${mturk.get_account_balance()['AvailableBalance']} in MTurk Account")

    assignment_df = pandas.read_csv(args["assignments_file"])

    assignment_results = []
    for i, row in assignment_df.iterrows():

        hit_id = row["hit_id"]
        print(f"Retrieve HIT: {hit_id}")

        # Get a list of the Assignments that have been submitted
        response = mturk.list_assignments_for_hit(
            HITId=str(hit_id),
            AssignmentStatuses=['Submitted', 'Approved'],
            MaxResults=100
        )

        assignments = response['Assignments']

        for assignment in assignments:

            item = {}

            answer_dict = xmltodict.parse(assignment['Answer'])
            answer_dict_2 = answer_dict["QuestionFormAnswers"]["Answer"]
            answer_processed_dict = {}
            for answer_pair in answer_dict_2:
                field_name = answer_pair["QuestionIdentifier"]
                field_value = answer_pair["FreeText"]
                answer_processed_dict[field_name] = field_value
            del(assignment['Answer'])

            item = {**item, **assignment}
            item = {**item, **answer_processed_dict}

            assignment_results.append(item)

    assignments_df = pandas.DataFrame(data=assignment_results)
    print(assignments_df)
    assignments_df.to_csv(args["output_file"])


create_assignments(vars(args))
