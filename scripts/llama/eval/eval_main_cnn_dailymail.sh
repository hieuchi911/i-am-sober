#! /bin/bash

NPROCS=2
MODEL_PARALLEL_SIZE=2
DISTRIBUTED_ARGS="--nproc_per_node ${NPROCS} \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 2012"

# BASE_PATH="/home/zihaoh/repos/i-am-sober"
BASE_PATH="/home1/hieutn/cs566/i-am-sober"
WANDB_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
WANDB_PRJ="i_am_sober"

# model
MODEL_NAME="llama-8B-selfkd-baseline"
# MODEL_PATH="/scratch1/hieutn/test-selfkd/llama/e3-bs16-lr1e-05-G1-N16-NN1-kd1.0-mp4/702/"
MODEL_PATH="/project/lerman_316/hieutn/ckps-lf/llama3-8b/full/sft/checkpoint-2814/"
MODEL_TYPE="llama"
# data
DATA_NAMES="cnn_dailymail"
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}/${MODEL_TYPE}
DATA_DIR="/scratch1/hieutn/test-selfkd/pseudo-data/"
# hp
EVAL_BS=4
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/eval_main_selfkd_pseudo_baseline_2814/"
TYPE="eval_main"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --ckpt-name ${MODEL_NAME}"
OPTS+=" --n-gpu ${NPROCS}"
OPTS+=" --model-type llama"
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MODEL_PARALLEL_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BS}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type ${TYPE}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
