#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH --account=lerman_316
#SBATCH --partition=gpu
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=8

#SBATCH --job-name=multinode
#SBATCH --account=lerman_316
#SBATCH --partition=gpu
#SBATCH --mem=248GB
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=2
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1


source ~/.bashrc

conda activate sober

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

NPROCS_PER_NODE=2
NNODES=8
NPROCS=$((NNODES * NPROCS_PER_NODE))
MODEL_PARALLEL_SIZE=4

# torchrun
SRUN_ARGS="--nodes=${NNODES} --ntasks-per-node=1"
DISTRIBUTED_ARGS="--nproc_per_node ${NPROCS_PER_NODE} \
                  --nnodes ${NNODES} \
                  --rdzv_id $RANDOM \
                  --rdzv_backend c10d \
                  --rdzv_endpoint $head_node_ip:29603"

BASE_PATH="/home1/hieutn/cs566/i-am-sober"
WANDB_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
WANDB_PRJ="i_am_sober_self_kd"

# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"  # path to model snapshots
MODEL_NAME="llama-8B-Student"
TEACHER_PATH="/scratch1/hieutn/test-sft/"    # path to model snapshots
TEACHER_MODEL_NAME="llama-8B-sft-Teacher"
MODEL_TYPE="llama"
# hp
BS=16
EVAL_BS=32
EPOCHS=3
LR=1e-05
GRAD_ACC=1
KD_RATIO=1.0
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
# DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/pseudo
TASK="summ"
# runtime
# SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/kd"
SAVE_PATH="/scratch1/hieutn/selfkd/${MODEL_TYPE}/"
# seed
SEED=10
SEED_ORDER=10

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
OPTS+=" --task ${TASK}"
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
export WANDB_SILENT=1
export WANDB_API_KEY=${WANDB_KEY}
export WANDB_PROJECT=${WANDB_PRJ}
export WANDB_NAME="kd-${MODEL_TYPE}-lr${LR}_bs${BS}_kd${KD_RATIO}"

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="srun ${SRUN_ARGS} torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
${CMD}
























BASE_PATH="/home1/hieutn/cs566/i-am-sober" # path to i-am-sober folder
# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"  # path to model snapshots
MODEL_TYPE="llama"
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
TRAIN_SIZE=15000
DEV_NUM=1000

echo "PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_cnn_dailymail.py \
--model-path ${MODEL_PATH} \
--model-type ${MODEL_TYPE} \
--hugg-data-id abisee/cnn_dailymail \
--hugg-data-subset 1.0.0 \
--max-length ${MAX_LENGTH} \
--max-prompt-length ${MAX_PROMPT_LENGTH} \
--processed-data-dir ${DATA_DIR} \
--data-process-workers 32 \
--train-size ${TRAIN_SIZE} \
--dev-num ${DEV_NUM}"

# Tokenize data and save in binary files
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_cnn_dailymail.py \
    --hugg-data-id abisee/cnn_dailymail \
    --hugg-data-subset 1.0.0 \
    --processed-data-dir ${DATA_DIR} \
    --model-path ${MODEL_PATH} \
    --model-type ${MODEL_TYPE} \
    --data-process-workers 32 \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --train-size ${TRAIN_SIZE} \
    --dev-num ${DEV_NUM}