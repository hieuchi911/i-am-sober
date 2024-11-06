# #!/bin/bash

# #SBATCH --job-name=multinode
# #SBATCH --account=lerman_316
# #SBATCH --partition=gpu
# #SBATCH --mem=150GB
# #SBATCH --cpus-per-task=32
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:v100:2
# #SBATCH --nodes=6
# #SBATCH --ntasks-per-node=1

# source ~/.bashrc
# conda activate sober-v1

# # system configs==============================================================
# NNODES=$SLURM_NNODES
# GPUS_PER_NODE=2
# NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# # so processes know who to talk to
# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

MASTER_PORT=29603

ACCELERATE_CONFIG="/scratch1/hieutn/accelerate/default_config.yaml" # rank 0

# export LAUNCHER="accelerate launch \
#     --config_file $ACCELERATE_CONFIG \
#     --main_process_ip $head_node_ip \
#     --main_process_port $MASTER_PORT \
#     --machine_rank \$SLURM_PROCID \
#     --num_processes $NUM_PROCESSES \
#     --num_machines $NNODES \
#     "


export LAUNCHER="accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --main_process_ip 10.125.0.217 \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes 2 \
    --num_machines 1 \
    "

# finetuning configs==========================================================
BASE_PATH="/home1/hieutn/cs566/i-am-sober"
# BASE_PATH="/home/zihaoh/repos/i-am-sober"
# ACCELERATE_CONFIG="/home/zihaoh/repos/i-am-sober/configs/accelerate.yml"

# data
DATA_DIR="/home1/hieutn/cs566/i-am-sober/processed_data/dolly/full-1024-512/llama3.1/"
# DATA_DIR="${BASE_PATH}/processed_data/dolly/full-1024-512/llama3.1/"

# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
# MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393"
# MODEL_PATH="/home/shared/transformers_cache/hub/<llama-70B-snapshots>"

# runtime
SAVE_PATH="${BASE_PATH}/results/llama3.1/train/sft"

# hp
BS=2
EVAL_BS=4

# Program:
export PROGRAM="\
${BASE_PATH}/finetune_v1.py \
    --base-path $BASE_PATH \
    --model-path $MODEL_PATH \
    --ckpt-name llama-70B-teacher \
    --model-type llama3.1 \
    --gradient-checkpointing \
    --data-dir $DATA_DIR \
    --num-workers 1 \
    --dev-num -1 \
    --lr 5e-06 \
    --batch-size $BS \
    --eval-batch-size $EVAL_BS \
    --gradient-accumulation-steps 1 \
    --warmup-iters 0 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --epochs 3 \
    --max-length 1024 \
    --max-prompt-length 512 \
    --do-train \
    --do-valid \
    --eval-gen \
    --save-interval -1 \
    --eval-interval -1 \
    --log-interval 4 \
    --mid-log-num -1 \
    --save $SAVE_PATH \
    --seed 10 \
    --seed-order 10 \
    --type lm \
    --do-sample \
    --top-k 0 \
    --top-p 1.0 \
    --temperature 1.0 \
    --peft lora \
    --peft-lora-r 8
"
# Other environment vars======================================================
# wandb
export WANDB_API_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
export WANDB_PROJECT="i_am_sober_dolly_sft_llama3.1"
export WANDB_NAME="sft-teacher-llama3.1-70B-lora-accelerate"
LOG_PATH="main_log.txt"

# Execute experiment==========================================================
export CMD="$LAUNCHER $PROGRAM"
echo $CMD
# $CMD
# srun --jobid $SLURM_JOBID bash -c "echo $CMD"
# srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
