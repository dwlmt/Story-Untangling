import argparse
import multiprocessing
import os

import dask
import dask.bag
import dask.array
import faiss
import hdbscan
import numpy
import pandas
from jsonlines import jsonlines
from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.preprocessing import normalize
from tqdm import tqdm
import umap

''' This is  hacky script. Sometimes for clustering the csv fails but the clustering succeeds, so just rerun the last part.
'''
parser = argparse.ArgumentParser(
    description='Load parquet file and save the summary variables.')
parser.add_argument('--vectors', required=True, type=str, help="Vector file to load.")
parser.add_argument('--output', required=True, type=str, help="Output to save the CSV")
parser

args = parser.parse_args()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)

def save_to_csv(args):
    print(args)

    ensure_dir(args["output"])

    import dask.dataframe as dd

    original_df = dd.read_parquet(args['vectors'])
    columns = [c for c in list(original_df.columns) if not c.endswith("diff") and not c.endswith("tensor") and not c.endswith("48")]
    csv_df = original_df[columns]

    csv_df = csv_df.compute()
    print(csv_df)
    csv_df.to_csv(f'{args["output"]}/vectors.csv.xz')

save_to_csv(vars(args))