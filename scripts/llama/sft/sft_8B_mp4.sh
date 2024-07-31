#! /bin/bash
NPROCS=2 # number of GPUs to use
MODEL_PARALLEL_SIZE=2

BASE_PATH="/home1/hieutn/cs566/i-am-sober" # path to i-am-sober folder
WANDB_KEY="<WANDB-API-KEY>"
WANDB_PRJ="i_am_sober"

# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"  # path to model snapshots
MODEL_NAME="llama-8B-baseline"
MODEL_TYPE="llama"
# hp
BS=2
EVAL_BS=2
EPOCHS=3
LR=0.00001
GRAD_ACC=1
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
TASK="summ"
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/sft"
SAVE_INTERVAL=-1
# seed
SEED=10
SEED_ORDER=10

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
        --model_type) MODEL_TYPE=$2; shift ;;
        --data_dir) DATA_DIR=$2; shift ;;
        --task) TASK=$2; shift ;;
        --lr) LR=$2; shift ;;
        --bs) BS=$2; shift ;;
        --eval_bs) EVAL_BS=$2; shift ;;
        --epochs) EPOCHS=$2; shift ;;
        --grad_acc) GRAD_ACC=$2; shift ;;
        --max_length) MAX_LENGTH=$2; shift ;;
        --max_prompt_length) MAX_PROMPT_LENGTH=$2; shift ;;
        --save_path) SAVE_PATH=$2; shift ;;
        --save_interval) SAVE_INTERVAL=$2; shift ;;
        --seed) SEED=$2; shift ;;
        --seed_order) SEED_ORDER=$2; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

DISTRIBUTED_ARGS="--nproc_per_node ${NPROCS} \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 2012"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --ckpt-name ${MODEL_NAME}"
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

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval ${SAVE_INTERVAL}"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"

# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"

# type
OPTS+=" --type lm"

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
export WANDB_NAME="sft-${MODEL_TYPE}-lr${LR}_bs${BS}"

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
echo ${WANDB_API_KEY}
echo ${WANDB_PROJECT}
echo ${WANDB_NAME}
echo "==========="
${CMD}