import argparse

import boto3
import pandas

parser = argparse.ArgumentParser(
    description='Approve and reject MTurk assignments.')
parser.add_argument('--mturk-results-file', required=True, type=str, help="Link to the annotations file and codes.")
parser.add_argument('--access-key-id', required=False, type=str, default="AKIAI33ZD245LOFYMYAQ", help="AWS Access Key.")
parser.add_argument('--secret-access-key', required=False, type=str, default="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",
                    help="AWS Access Key.")
parser.add_argument('--mturk-url', required=False, type=str,
                    default="https://mturk-requester-sandbox.us-east-1.amazonaws.com/", help="Mechanical turk link.")

args = parser.parse_args()

def approve_reject_assignments(args):
    print(f"Create assignments: {args}")

    mturk = boto3.client('mturk',
                         aws_access_key_id=args["access_key_id"],
                         aws_secret_access_key=args["secret_access_key"],
                         region_name='us-east-1',
                         endpoint_url=args["mturk_url"],
                         )
    print(f"I have $ ${mturk.get_account_balance()['AvailableBalance']} in MTurk Account")

    assignment_df = pandas.read_csv(args["mturk_results_file"])

    for i, row in assignment_df.iterrows():

        try:
            if "Approve" in row and row["Approve"] == True:
                print(f"Approve assignment id: {row['AssignmentId']}, for worker {row['WorkerId']} and HIT {row['HITId']}")
                mturk.approve_assignment(AssignmentId=row['AssignmentId'])
            elif "Reject" in row and row["Reject"] == True:
                print(
                    f"Reject assignment id: {row['AssignmentId']}, for worker {row['WorkerId']} and HIT {row['HITId']} for reason - {row['RejectReason']}")
                mturk.reject_assignment(AssignmentId=row['AssignmentId'], RequesterFeedback=row['RejectReason'])
        except:
            pass


# approve_assignment(assignment_id, feedback=None)
approve_reject_assignments(vars(args))
