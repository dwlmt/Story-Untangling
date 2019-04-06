import argparse
import collections
import datetime
import locale
from collections import defaultdict

import numpy
import pandas
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import interval_distance
from scipy.stats import stats

locale.setlocale(locale.LC_TIME, "en_US")  # swedish

genre_categories = ['Answer.crime.on',
                    'Answer.erotic_fiction.on', 'Answer.fable.on', 'Answer.fairytale.on',
                    'Answer.fan_fiction.on', 'Answer.fantasy.on', 'Answer.folktale.on',
                    'Answer.historical_fiction.on', 'Answer.horror.on', 'Answer.humor.on',
                    'Answer.legend.on', 'Answer.magic_realism.on', 'Answer.meta_fiction.on',
                    'Answer.mystery.on', 'Answer.mythology.on', 'Answer.mythopoeia.on',
                    'Answer.other.on',
                    'Answer.realistic_fiction.on', 'Answer.science_fiction.on',
                    'Answer.swashbuckler.on', 'Answer.thriller.on']

story_id_col = 'Answer.storyId'

worker_id_col = 'WorkerId'

stats_columns = ['Answer.doxaResonance',
                 'Answer.doxaSurprise', 'Answer.doxaSuspense',
                 'Answer.readerEmotionalResonance',
                 'Answer.readerSurprise', 'Answer.readerSuspense',
                 'Answer.storyInterest', 'Answer.storySentiment']

genre_column = "genre"


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def process_annotations(args):
    source_csv = pandas.read_csv(args["source_csv"])

    accept_time_col = source_csv["AcceptTime"]
    submit_time_col = source_csv["SubmitTime"]

    # Highlight those that are too short.
    suspiciously_quick = []
    for accept_time, submit_time in zip(accept_time_col, submit_time_col):
        accept_time = accept_time.replace("PDT", "").strip()
        submit_time = submit_time.replace("PDT", "").strip()

        mturk_date_format = "%a %b %d %H:%M:%S %Y"
        accept_time = datetime.datetime.strptime(accept_time, mturk_date_format)
        submit_time = datetime.datetime.strptime(submit_time, mturk_date_format)

        time_taken = submit_time - accept_time

        if time_taken.seconds / 60.0 < args["min_time"]:
            suspiciously_quick.append(True)
        else:
            suspiciously_quick.append(False)
    source_csv = source_csv.assign(too_quick=pandas.Series(suspiciously_quick))

    # Story summary
    token_length = []
    too_short = []
    for summary in source_csv["Answer.storySummary"]:
        num_tokens = len(summary.split(" "))
        token_length.append(num_tokens)
        if num_tokens < args["min_tokens"]:
            too_short.append(True)
        else:
            too_short.append(False)

    source_csv = source_csv.assign(num_summary_tokens=pandas.Series(token_length))
    source_csv = source_csv.assign(too_short=pandas.Series(too_short))

    genres = []
    for index, row in source_csv.iterrows():
        added = False
        for g in genre_categories:
            if row[g] == True and not added:
                added = True
                genre_name = g.split(".")[1]
                genres.append(genre_name)
        if not added:
            genres.append("other")
    source_csv = source_csv.assign(genre=pandas.Series(genres))

    source_csv.to_csv(f"{args['target']}_processed.csv")

    stats_dict = defaultdict(dict)
    for col in stats_columns:
        figures = source_csv[col]

        nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(figures)

        stats_dict[col]["nobs"] = nobs
        stats_dict[col]["min"] = minmax[0]
        stats_dict[col]["max"] = minmax[1]
        stats_dict[col]["mean"] = mean
        stats_dict[col]["variance"] = variance
        stats_dict[col]["skewness"] = skewness
        stats_dict[col]["kurtosis"] = kurtosis

        stats_dict[col]["25_perc"] = numpy.percentile(figures, 25)
        stats_dict[col]["median"] = numpy.percentile(figures, 50)
        stats_dict[col]["75_perc"] = numpy.percentile(figures, 75)

        triples = []
        for index, row in source_csv.iterrows():
            worker = row[worker_id_col]
            story = row[story_id_col]
            metrics_col = row[col]
            triples.append((worker, story, metrics_col))
        t = AnnotationTask(data=triples, distance=interval_distance)
        stats_dict[col]["krippendorff_alpha"] = t.alpha()

    genre_desc_count = source_csv[genre_column].value_counts(normalize=False)
    genre_desc = source_csv[genre_column].value_counts(normalize=True)
    for (n, v), (nc, vc) in zip(genre_desc.iteritems(), genre_desc_count.iteritems()):
        stats_dict[genre_column][n] = {"count": vc, "proportion": v}

    flattened_stats = flatten(stats_dict, sep=".")

    pandas.DataFrame.from_dict(flattened_stats, orient="index").to_csv(f"{args['target']}_stats.csv")

    corr_cov_df = source_csv[stats_columns]

    for method in ('pearson', 'kendall', 'spearman'):
        correlation_df = corr_cov_df.corr(method=method)
        correlation_df.to_csv(f"{args['target']}_{method}_corr.csv")
    covariance_df = corr_cov_df.cov()

    covariance_df.to_csv(f"{args['target']}_cov.csv")

    print(source_csv.columns)


parser = argparse.ArgumentParser(
    description='Process the whole corpus annotation data, convert values and calculate ')
parser.add_argument('--source-csv', required=True, type=str, help="The RAW Amazon Mechanical Turk ")
parser.add_argument('--target', required=True, type=str, help="Base output name for the saved statistics.")
parser.add_argument('--min-time', type=int, default=5, help="Min time in minutes not to be considered suspicious.")
parser.add_argument('--min-tokens', type=int, default=5, help="Min tokens considered suspicious.")

args = parser.parse_args()

process_annotations(vars(args))
