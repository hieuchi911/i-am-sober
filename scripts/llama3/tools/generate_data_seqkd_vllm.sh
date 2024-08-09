#! /bin/bash

NPROCS=2

# model
BASE_PATH="/home/zihaoh/repos/i-am-sober" # path to i-am-sober folder
MODEL_PATH="/scratch1/hieutn/test-sft/" # path to SFT model
MODEL_NAME="llama-8B-sft"
MODEL_TYPE="llama3"
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_NAME="cnn_dailymail"
DATA_DIR=${BASE_PATH}/processed_data/${DATA_NAME}/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}/${MODEL_TYPE}/
# hp
EVAL_BS=64
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/gen/"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --ckpt-name ${MODEL_NAME}"
OPTS+=" --n-gpu ${NPROCS}"
OPTS+=" --model-type ${MODEL_TYPE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --num-workers 0"
OPTS+=" --gen-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BS}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --type gen"
OPTS+=" --vllm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 1"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="python ${BASE_PATH}/generate.py ${OPTS} $@"


echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}