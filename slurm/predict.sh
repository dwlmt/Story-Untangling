#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

# Activate Conda
source /home/${USER}/miniconda3/bin/activate pytorch

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M');
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

if [ -d "/disk/scratch1" ]; then
export SCRATCH_ROOT=/disk/scratch1/s1569885/
elif [ -d "/disk/scratch2" ]; then
 export SCRATCH_ROOT=/disk/scratch2/s1569885/
elif [ -d "/disk/scratch" ]; then
 export SCRATCH_ROOT=/disk/scratch/s1569885/
else
 export SCRATCH_ROOT=/disk/scratch_big/s1569885/
fi

export SCRATCH_HOME="/${SCRATCH_ROOT}/${STUDENT_ID}"

export CLUSTER_HOME="/home/${STUDENT_ID}"
export NLTK_DATA="${CLUSTER_HOME}/nltk_data/"
export EXP_ROOT="${CLUSTER_HOME}/git/Story-Untangling/"
export ALLENNLP_CACHE_ROOT="${CLUSTER_HOME}/allennlp_cache_root/"

export SERIAL_DIR="${SCRATCH_HOME}/suspense_acl20/${EXP_NAME}"

# Predictor specific variables.
export DATASET_PATH="/home/s1569885/comics/stories/WritingPrompts/dataset_db/text/"
export PREDICTION_STORY_ID_FILE="/home/s1569885/comics/stories/WritingPrompts/annotation_results/raw/story_id_test_1.csv"
export PREDICTION_ONLY_ANNOTATION_STORIES=TRUE
export PREDICTION_LEVELS_TO_ROLLOUT=1
export PREDICTION_GENERATE_PER_BRANCH=100
export PREDICTION_SAMPLE_PER_BRANCH=100
export PREDICTION_BEAM_SIZE=10
export PREDICTION_SAMPLE_TOP_K_WORDS=50
export PREDICTION_WINDOWING=TRUE
export PREDICTION_SENTIMENT_WEIGHTING=1.0
export PREDICTION_SENTIMENT_POSITIVE_WEIGHTING=1.0
export PREDICTION_SENTIMENT_NEGATIVE_WEIGHTING=1.0
export PREDICTION_MARKED_SENTENCE=FALSE

# Ensure the scratch home exists and CD to the experiment root level.
mkdir -p "${SCRATCH_HOME}"
cd "${EXP_ROOT}" # helps AllenNLP behave

mkdir -p ${SERIAL_DIR}

echo "ALLENNLP Task========"

allennlp predict --include-package story_untangling \
    --use-dataset-reader \
    --predictor uncertain_reader_gen_predictor \
     ${CLUSTER_HOME}/comics/stories/WritingPrompts/training_models/full_epoch/lstm_fusion_big/ \
     ${CLUSTER_HOME}/comics/stories/WritingPrompts/datasets/test_wp.target --cuda-device 0 \
    --output-file  ${SERIAL_DIR}/${EXP_NAME}_prediction_output.jsonl \

echo "============"
echo "ALLENNLP Task finished"

rsync -avuzhP "${SERIAL_DIR}" "${EXP_ROOT}/runs/cluster/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}/"

echo "============"
echo "results synced"
