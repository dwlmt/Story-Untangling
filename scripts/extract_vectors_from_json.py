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
from tqdm import tqdm
import umap

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--source-json', required=True, type=str, help="JSON file to process.")
parser.add_argument('--output', required=True, type=str, help="Output, saved as a Parquet file")
parser.add_argument('--umap-n-neighbours', default=15, type=int, help="The number of neighbours.")
parser.add_argument('--umap-min-dist', default=0.1, type=float, help="Controls how clumpy umap will cluster points.")
parser.add_argument('--similarity-metric', default=["euclidean"], nargs="+", type=str, help="The similarity measure to use.")
parser.add_argument('--dim-reduction-components', default=[48, 2], type=int, nargs="+", help="The number of components to reduce to.")
parser.add_argument('--min-cluster-size', default=5, type=int, help="Min size fo each cluster.")
parser.add_argument('--kmeans-ncentroids', default=32, type=int, help="Number of K-means centroids.")
parser.add_argument('--kmeans-iterations', default=20, type=int, help="Number of K-means iteration.")
parser.add_argument('--code-size', default=4, type=int, help="Byte size for the product quantization")
parser.add_argument('--num-stories', default=100, type=int, help="Max number of stories to process")
parser.add_argument("--no-hdbscan", default=False, action="store_true" , help="Don't run HDBSCAN")
parser.add_argument("--no-kmeans", default=False, action="store_true" , help="Don't run K-Means and the product code.")
parser.add_argument("--no-umap", default=False, action="store_true" , help="Don't run UMAP dim reduction.")
parser.add_argument("--no-pca", default=False, action="store_true" , help="Don't run PCA dim reduction.")
parser.add_argument("--save-csv", default=False, action="store_true" , help="Save to CSV files as well as Parquet.")
parser.add_argument('--gpus', default=4, type=int, help="GPUs")


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)

args = parser.parse_args()

def train_kmeans(x, k, niter, ngpu):
    "Runs kmeans on one or several GPUs"
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = niter

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return centroids.reshape(k, d)

def extract_json_stats(args):
    print(args)

    vector_fields = ["sentence_tensor","story_tensor"]
    metadata_fields = ["story_id", "sentence_id", "sentence_num", "sentence_length", "sentence_text"]

    ensure_dir(args["output"])

    original_df = dask.bag.from_sequence(extract_rows(args, metadata_fields, vector_fields, args["num_stories"]))
    original_df = original_df.to_dataframe()


    cluster_dim_fields = []
    for vector_field in vector_fields:

        if not args["no_umap"]:
            vector_data = numpy.stack(numpy.array(original_df[vector_field].compute().values), axis=0)

            for sim_metric in args["similarity_metric"]:
                for dim in args["dim_reduction_components"]:

                    new_vector_field = f"{vector_field}_{sim_metric}_umap_{dim}"

                    print(f"Dimensionality reduction for: {new_vector_field}")

                    vector_data = umap.UMAP(n_neighbors=args["umap_n_neighbours"], min_dist=args["umap_min_dist"], n_components=dim, metric=sim_metric).fit_transform(
                        vector_data)

                    if dim > 3:
                        cluster_dim_fields.append(new_vector_field)

                    source = []
                    for sentence_id, reduced_dim in zip (original_df["sentence_id"].compute().values.tolist(),numpy.split(vector_data, 1)[0]):
                        source.append({"sentence_id": sentence_id, new_vector_field : reduced_dim})

                    dim_df = dask.bag.from_sequence(source).to_dataframe()

                    original_df = original_df.merge(dim_df, left_on="sentence_id", right_on="sentence_id")

        if not args["no_pca"]:
            vector_data = numpy.stack(numpy.array(original_df[vector_field].compute().values), axis=0)

            # PCA Dim reduction.
            for dim in args["dim_reduction_components"]:

                new_vector_field = f"{vector_field}_pca_{dim}"
                print(f"Dimensionality reduction for: {new_vector_field}")

                mat = faiss.PCAMatrix(vector_data.shape[1], dim)
                mat.train(vector_data)

                assert mat.is_trained
                vector_data = mat.apply_py(vector_data)

                if dim > 3:
                    cluster_dim_fields.append(new_vector_field)

                source = []
                for sentence_id, reduced_dim in zip(original_df["sentence_id"].compute().values.tolist(),
                                                    numpy.split(vector_data, 1)[0]):
                    source.append({"sentence_id": sentence_id, new_vector_field: reduced_dim})
                dim_df = dask.bag.from_sequence(source).to_dataframe()

            original_df = original_df.merge(dim_df, left_on="sentence_id", right_on="sentence_id")

    if not args["no_kmeans"]:
        res = faiss.StandardGpuResources()

        for field in cluster_dim_fields:

            cluster_field = f"{field}_cluster"
            print(f"K-Means Clustering for : {cluster_field}")

            vector_data = numpy.stack(numpy.array(original_df[field].compute().values), axis=0)

            d = vector_data.shape[1]

            pq = faiss.ProductQuantizer(d, args["code_size"], 8)

            opq = faiss.OPQMatrix(d, args["code_size"])
            opq.pq = pq
            opq.train(vector_data)
            xt = opq.apply_py(vector_data)

            codes = pq.compute_codes(xt)

            ncentroids = args["kmeans_ncentroids"]
            niter = args["kmeans_iterations"]

            centroids = train_kmeans(vector_data, ncentroids, niter, args["gpus"])

            pandas.DataFrame(centroids).to_csv(f'{args["output"]}_{field}_centroids.csv')

            centroid_index = faiss.GpuIndexFlatL2(res, d)
            centroid_index.add(centroids)

            D, I = centroid_index.search(vector_data, 1) #kmeans.index.search(vector_data, 1)

            source = []
            for sentence_id, distance, cluster, code in zip(original_df["sentence_id"].compute().values.tolist(), D, I,
                                                            codes):
                source.append({"sentence_id": sentence_id, f"{cluster_field}_kmeans_distance": distance[0],
                               f"{cluster_field}_kmeans_cluster": cluster[0], f"{cluster_field}_product_code": code})
                print(source[-1])

            dim_df = dask.bag.from_sequence(source).to_dataframe()

            original_df = original_df.merge(dim_df, left_on="sentence_id", right_on="sentence_id")

    if not args["no_hdbscan"]:
        for field in cluster_dim_fields:
            for sim_metric in args["similarity_metric"]:

                if sim_metric not in field:
                    continue

                cluster_field = f"{field}_cluster"
                print(f"HDBSCAN Clustering for : {cluster_field}")

                clusterer = hdbscan.HDBSCAN(algorithm='best', metric=sim_metric,
                                            min_cluster_size=args["min_cluster_size"], core_dist_n_jobs=multiprocessing.cpu_count() - 1)

                vector_data = dask.array.from_array(numpy.array(original_df[field].compute().values.tolist()), chunks=(1000, 1000))

                clusterer.fit(vector_data)

                source = []
                for sentence_id, label, prob, outlier in zip(original_df["sentence_id"].compute().values.tolist(),clusterer.labels_,
                                                             clusterer.probabilities_, clusterer.outlier_scores_):

                    source.append({"sentence_id": sentence_id, f"{cluster_field}_label": label,
                                   f"{cluster_field}_probability": prob, f"{cluster_field}_outlier_score": outlier})

                    print(source[-1])

                dim_df = dask.bag.from_sequence(source).to_dataframe()

                original_df = original_df.merge(dim_df, left_on="sentence_id", right_on="sentence_id")

    original_df.to_parquet(f'{args["output"]}', compression='snappy')
    if args["save_csv"]:
        original_df = original_df.compute(scheduler='processes')
        print(original_df)
        original_df.to_csv(f'{args["output"]}.csv')

def extract_rows(args, metadata_fields, vector_fields, max_num_stories):
    index_counter = 0
    with jsonlines.open(args['source_json']) as reader:
        for i, obj in tqdm(enumerate(reader)):
            for child in obj["children"]:

                yield {**{k: child[k] for k in metadata_fields},  **{k: numpy.array(child[k],dtype=numpy.float32) for k in vector_fields}}

                index_counter += 1

            if i == max_num_stories:
                break


extract_json_stats(vars(args))