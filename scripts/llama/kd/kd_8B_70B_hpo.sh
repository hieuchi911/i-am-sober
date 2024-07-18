#!/bin/bash
NPROCS=2 # number of GPUs to use
MODEL_PARALLEL_SIZE=2
BASE_PATH="/home1/hieutn/cs566/i-am-sober" # path to i-am-sober folder
WANDB_KEY="<WANDB-API-KEY>"
WANDB_PRJ="i_am_sober"
# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"  # path to model snapshots
MODEL_NAME="llama-8B-Student"
TEACHER_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"    # path to model snapshots
TEACHER_MODEL_NAME="llama-70B-Teacher"
MODEL_TYPE="llama"
# hp
LR=(5e-5 1e-5 5e-6)
BS=(8 4)
KD_RATIO=(0.1 1.0 10)
EVAL_BS=8
EPOCHS=3
GRAD_ACC=1
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
TASK="summ"
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/kd"
# seed
SEED=10
SEED_ORDER=10

# run scripts/llama/kd/kd_8B_70B.sh with all combinations of hyperparameters
for l in ${LR[@]}; do
  for b in ${BS[@]}; do
    for r in ${KD_RATIO[@]}; do
      bash scripts/llama/kd/kd_8B_70B.sh --nprocs ${NPROCS} --model_parallel_size ${MODEL_PARALLEL_SIZE} \
           --base_path ${BASE_PATH} --wandb_key ${WANDB_KEY} --wandb_prj ${WANDB_PRJ} --model_path ${MODEL_PATH} \
           --model_name ${MODEL_NAME} --teacher_path ${TEACHER_PATH} --teacher_model_name ${TEACHER_MODEL_NAME} \
           --model_type ${MODEL_TYPE} --data_dir ${DATA_DIR} --task ${TASK} --lr ${l} --bs ${b} --kd_ratio ${r} \
           --eval_bs ${EVAL_BS} --epochs ${EPOCHS} --grad_acc ${GRAD_ACC} --max_length ${MAX_LENGTH} \
           --max_prompt_length ${MAX_PROMPT_LENGTH} --save_path ${SAVE_PATH} --seed ${SEED} \
           --seed_order ${SEED_ORDER}
    done
  done
done
