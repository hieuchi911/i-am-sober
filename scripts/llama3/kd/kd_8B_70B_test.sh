#!/bin/bash
NPROCS=2 # number of GPUs to use
MODEL_PARALLEL_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node ${NPROCS} \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 2012"

BASE_PATH="/home1/hieutn/cs566/i-am-sober" # path to i-am-sober folder
WANDB_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
WANDB_PRJ="i_am_sober"

# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"  # path to model snapshots
MODEL_NAME="llama-8B-Student"
TEACHER_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"    # path to model snapshots
TEACHER_MODEL_NAME="llama-70B-Teacher"
MODEL_TYPE="llama"

# MODEL_PATH="/scratch1/hieutn/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62/"  # path to model snapshots
# MODEL_NAME="opt-1.3b-Student"
# TEACHER_PATH="/scratch1/hieutn/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62/"    # path to model snapshots
# TEACHER_MODEL_NAME="opt-1.3b-Teacher"
# MODEL_TYPE="opt"
# data
DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
# hp
BS=8
EVAL_BS=8
EPOCHS=3
LR=0.00001
GRAD_ACC=1
KD_RATIO=1
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/kd"
# seed
SEED=10
SEED_ORDER=10

# HPO
# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nprocs) NPROCS=$2; shift ;;
        --model_parallel_size) MODEL_PARALLEL_SIZE=$2; shift ;;
        --base_path) BASE_PATH=$2; shift ;;
        --wandb_key) WANDB_KEY=$2; shift ;;
        --wandb_prj) WANDB_PRJ=$2; shift ;;
        --model_path) MODEL_PATH=$2; shift ;;
        --model_name) MODEL_NAME=$2; shift ;;
        --teacher_path) TEACHER_PATH=$2; shift ;;
        --teacher_model_name) TEACHER_MODEL_NAME=$2; shift ;;
        --model_type) MODEL_TYPE=$2; shift ;;
        --data_dir) DATA_DIR=$2; shift ;;
        --bs) BS=$2; shift ;;
        --lr) LR=$2; shift ;;
        --kd_ratio) KD_RATIO=$2; shift ;;
        --eval_bs) EVAL_BS=$2; shift ;;
        --epochs) EPOCHS=$2; shift ;;
        --grad_acc) GRAD_ACC=$2; shift ;;
        --max_length) MAX_LENGTH=$2; shift ;;
        --max_prompt_length) MAX_PROMPT_LENGTH=$2; shift ;;
        --save_path) SAVE_PATH=$2; shift ;;
        --seed) SEED=$2; shift ;;
        --seed_order) SEED_ORDER=$2; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "NPROCS: ${NPROCS}"
echo "MODEL_PARALLEL_SIZE: ${MODEL_PARALLEL_SIZE}"
echo "BASE_PATH: ${BASE_PATH}"
echo "WANDB_KEY: ${WANDB_KEY}"
echo "WANDB_PRJ: ${WANDB_PRJ}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "TEACHER_PATH: ${TEACHER_PATH}"
echo "TEACHER_MODEL_NAME: ${TEACHER_MODEL_NAME}"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "DATA_DIR: ${DATA_DIR}"
echo "BS: ${BS}"
echo "LR: ${LR}"
echo "KD_RATIO: ${KD_RATIO}"
echo "EVAL_BS: ${EVAL_BS}"
echo "EPOCHS: ${EPOCHS}"
echo "GRAD_ACC: ${GRAD_ACC}"
echo "MAX_LENGTH: ${MAX_LENGTH}"
echo "MAX_PROMPT_LENGTH: ${MAX_PROMPT_LENGTH}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "SEED: ${SEED}"
echo "SEED_ORDER: ${SEED_ORDER}"
echo "========================================"
echo " "
# # Tokenize data and save in binary files
# PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_cnn_dailymail.py \
#     --hugg-data-id abisee/cnn_dailymail \
#     --hugg-data-subset 1.0.0 \
#     --processed-data-dir ${DATA_DIR} \
#     --model-path ${MODEL_PATH} \
#     --data-process-workers 32 \
#     --max-length ${MAX_LENGTH} \
#     --max-prompt-length ${MAX_PROMPT_LENGTH} \
#     --dev-num 1000 \
#     --model-type ${MODEL_TYPE}

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
OPTS+=" --gradient-checkpointing"
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MODEL_PARALLEL_SIZE}"

# data
OPTS+=" --data-dir ${DATA_DIR}/${MODEL_TYPE}/"
OPTS+=" --task summ"
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
OPTS+=" --save-interval 1000"
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
export WANDB_SILENT=1
export WANDB_API_KEY=${WANDB_KEY}
export WANDB_PROJECT=${WANDB_PRJ}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
# ${CMD}
