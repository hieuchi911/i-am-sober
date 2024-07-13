#!/bin/bash
NPROCS=${4-2} # number of GPUs to use
MODEL_PARALLEL_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node ${NPROCS} \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 2012"

BASE_PATH=${1-"/home1/hieutn/cs566/i-am-sober"} # path to i-am-sober folder
WANDB_KEY="<WANDB-API-KEY>"
WANDB_PRJ="i_am_sober"

# model
MODEL_PATH=${2-"/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"}  # path to model snapshots
MODEL_NAME="llama-8B-Student"
TEACHER_PATH=${3-"/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"}    # path to model snapshots
TEACHER_MODEL_NAME="llama-70B-Teacher"
MODEL_TYPE="llama"
# data
DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full
# hp
BS=8
EVAL_BS=8
EPOCHS=3
LR=0.00001
GRAD_ACC=1
KD_RATIO=0.5
# length
MAX_LENGTH=4096
MAX_PROMPT_LENGTH=3500
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/kd"
# seed
SEED=10
SEED_ORDER=10

# Tokenize data and save in binary files
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_cnn_dailymail.py \
    --hugg-data-id abisee/cnn_dailymail \
    --hugg-data-subset 1.0.0 \
    --processed-data-dir ${DATA_DIR} \
    --model-path ${MODEL_PATH} \
    --data-process-workers 32 \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --dev-num 1000 \
    --model-type ${MODEL_TYPE}

# # Change Model Parallel Size
# python tools/convert_mp.py --input_path ${MODEL_PATH} --source_mp_size 1 --target_mp_size ${NPROCS} --model_type ${MODEL_TYPE} --exist_ok
# python tools/convert_mp.py --input_path ${TEACHER_PATH} --source_mp_size 1 --target_mp_size ${NPROCS} --model_type ${MODEL_TYPE} --exist_ok

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --teacher-model-path ${TEACHER_PATH}"
OPTS+=" --ckpt-name ${MODEL_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_MODEL_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${NPROCS}"
OPTS+=" --model-type ${MODEL_TYPE}"
# OPTS+=" --gradient-checkpointing"
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MODEL_PARALLEL_SIZE}"

# data
OPTS+=" --data-dir ${DATA_DIR}/${MODEL_TYPE}/"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num -1"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BS}"
OPTS+=" --eval-batch-size ${EVAL_BS}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio ${KD_RATIO}"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"

# seed
OPTS+=" --seed ${SEED}"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# type
OPTS+=" --type kd"

# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=False
export WANDB_API_KEY=${WANDB_KEY}
export WANDB_PROJECT=${WANDB_PRJ}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
${CMD}
