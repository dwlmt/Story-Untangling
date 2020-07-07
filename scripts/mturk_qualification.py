import argparse
import datetime

import boto3
import pandas

parser = argparse.ArgumentParser(
    description='Grant a custom qualification on MTurk.')
parser.add_argument('--access-key-id', required=True, help="AWS Access Key.")
parser.add_argument('--secret-access-key', required=True, type=str, help="AWS Access Key.")
parser.add_argument('--mturk-url', required=False, type=str,
                    default="https://mturk-requester-sandbox.us-east-1.amazonaws.com/", help="Mechanical turk link.")
parser.add_argument('--workers', type=str, nargs="+", required=True, help="A list of workers to exclude from the task.")
parser.add_argument('--qualification-code', type=str, required=True, help="The MTurk Qualification Code to Grant and Revoke.")

subparsers = parser.add_subparsers(help='Specify grant or revoke.', dest='command')
grant_parser = subparsers.add_parser('grant', help='Grant the specified qualiication to the listed workers.')
revoke_parser = subparsers.add_parser('revoke', help='Revoke the qualification from the listed worker.')

args = parser.parse_args()

def grant_or_revoke_qualification(args):
    print(f"Amend qualifications: {args}")

    mturk = boto3.client('mturk',
                         aws_access_key_id=args["access_key_id"],
                         aws_secret_access_key=args["secret_access_key"],
                         region_name='us-east-1',
                         endpoint_url=args["mturk_url"],
                         )
    print(f"I have $ ${mturk.get_account_balance()['AvailableBalance']} in MTurk Account")

    for worker in args["workers"]:
        try:
            if args["command"] == "grant":
                result = mturk.associate_qualification_with_worker(QualificationTypeId=args["qualification_code"], WorkerId=worker, IntegerValue=1, SendNotification=False)
            elif args["command"] == "revoke":
                result = mturk.disassociate_qualification_with_worker(QualificationTypeId=args["qualification_code"], WorkerId=worker)
            print(result)
        except Exception as e:
            print(e)

grant_or_revoke_qualification(vars(args))
