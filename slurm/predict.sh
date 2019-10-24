#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 168:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

# Activate Conda
source /home/${USER}/miniconda3/bin/activate allennlp

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M');
echo ${dt}

# Env variables
export STUDENT_ID=${USER}
export SCRATCH_HOME="/disk/scratch/${STUDENT_ID}"
export CLUSTER_HOME="/home/${STUDENT_ID}"
export EXP_ROOT="${CLUSTER_HOME}/suspense/acl20"

export SERIAL_DIR="${SCRATCH_HOME}/suspense_acl20_exps/${EXP_NAME}"

# Ensure the scratch home exists and CD to the experiment root level.
mkdir -p "${SCRATCH_HOME}"
cd "${EXP_ROOT}" # helps AllenNLP behave

# Serialisation Directory for AllenNLP is the experiment folder
if [[ ! -f ${EXP_CONFIG} ]]; then
    echo "Config file ${EXP_CONFIG} is invalid, quitting..."
    exit 1
fi
mkdir -p ${SERIAL_DIR}

echo "ALLENNLP Task========"
allennlp train "${EXP_CONFIG}" \
    --serialization-dir "${SERIAL_DIR}" \
    --include-package story_untangling \

allennlp predict --include-package story_untangling \
    --use-dataset-reader \
    --predictor uncertain_reader_gen_predictor \
    /afs/inf.ed.ac.uk/group/project/comics/stories/WritingPrompts/training_models/movies_finetune/lstm_fusion_big/ \
    /afs/inf.ed.ac.uk/group/project/comics/TRIPOD_dataset/test.txt --cuda-device 0 \
    --output-file /afs/inf.ed.ac.uk/group/project/comics/stories/WritingPrompts/prediction_output/tripod/finetuned_models/cosine_100_lstm_test_exc_gold.jsonl \

echo "============"
echo "training finished successfully"

rsync -avuzhP "${SERIAL_DIR}" "${EXP_ROOT}/runs/cluster/" # Copy output onto headnode

echo "============"
echo "results synced"