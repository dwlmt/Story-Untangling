#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=0  # memory in Mb
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

# Activate Conda
source /home/${USER}/miniconda3/bin/activate pytorch

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M');
echo ${dt}

# Env variables
export STUDENT_ID=${USER}

declare -a ScratchPathArray=("/disk/scratch_big/" "/disk/scratch/" "/disk/scratch1" "/disk/scratch2/" "/disk/scratch_fast/" "{$CLUSTER_HOME}/scratch")
export SCRATCH_SPACE_REQUIRED=1000000000 # 1GB

# Iterate the string array using for loop
for i in "${ScratchPathArray[@]}"
do
   echo $i
   if [ -w "$i"] && [ `df "$i" | awk 'END{print $4}'` > $SCRATCH_SPACE_REQ ]; then
      export SCRATCH_ROOT="$i"
      break
   fi
done

export SCRATCH_HOME="/${SCRATCH_ROOT}/${STUDENT_ID}"

export NLTK_DATA="${CLUSTER_HOME}/nltk_data/"
export EXP_ROOT="${CLUSTER_HOME}/git/Story-Untangling/"
export ALLENNLP_CACHE_ROOT="${CLUSTER_HOME}/allennlp_cache_root/"

export SERIAL_DIR="${SCRATCH_HOME}/${EXP_NAME}"


# Predictor specific variables.
export DATASET_PATH="/home/s1569885/comics/stories/WritingPrompts/dataset_db/text/"
export PREDICTION_STORY_ID_FILE="/home/s1569885/comics/stories/WritingPrompts/annotation_results/raw/story_id_test_1.csv"
export PREDICTION_ONLY_ANNOTATION_STORIES=TRUE
export PREDICTION_LEVELS_TO_ROLLOUT=1
export PREDICTION_GENERATE_PER_BRANCH=100
export PREDICTION_SAMPLE_PER_BRANCH=100
export PREDICTION_BEAM_SIZE=5
export PREDICTION_SAMPLE_TOP_K_WORDS=50
export PREDICTION_WINDOWING=TRUE
export PREDICTION_SENTIMENT_WEIGHTING=1.0
export PREDICTION_SENTIMENT_POSITIVE_WEIGHTING=1.0
export PREDICTION_SENTIMENT_NEGATIVE_WEIGHTING=2.0
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
     ${CLUSTER_HOME}/comics/stories/WritingPrompts/datasets/test.wp_target --cuda-device 0 \
    --output-file  ${SERIAL_DIR}/${EXP_NAME}_prediction_output.jsonl \

echo "============"
echo "ALLENNLP Task finished"

mkdir -p "${CLUSTER_HOME}/runs/cluster/"
rsync -avuzhP "${SERIAL_DIR}" "${CLUSTER_HOME}/runs/cluster/" # Copy output onto headnode

rm -rf "${SERIAL_DIR}/"

echo "============"
echo "results synced"