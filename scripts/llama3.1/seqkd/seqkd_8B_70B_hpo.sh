#!/bin/bash
NPROCS=2 # number of GPUs to use
# BASE_PATH="/home/zihaoh/repos/i-am-sober" # path to i-am-sober folder
BASE_PATH="/home1/hieutn/cs566/i-am-sober" # path to i-am-sober folder

WANDB_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
WANDB_PRJ="i_am_sober_dolly_kd"
# model
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/" # path to model snapshots
MODEL_NAME="llama-8B-Student"
TEACHER_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/" # path to SFT teacher snapshots
TEACHER_MODEL_NAME="llama-70B-Teacher"
TEACHER_QUANTIZED="teacher_quantized"
TEACHER_LORA_PATH="mkopecki/chess-lora-adapter-llama-3.1-8b" # path to LoRA teacher

MODEL_TYPE="llama2"
# hp
LR=(1e-05) # (1e-05 5e-06)
BS=(8) # (8 4)
KD_RATIO=(0.1) # (0.1 1.0 10.0)
EVAL_BS=8
EPOCHS=3
GRAD_ACC=1
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_NAMES=(
  dolly general
  # cnn_dailymail summ
)
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/kd"
SAVE_INTERVAL=-1
# seed
SEED=10
SEED_ORDER=10

# run scripts/llama/kd/kd_8B_70B.sh with all combinations of hyperparameters
for ((i=0; i<${#DATA_NAMES[@]}; i+=2)); do
  DATA_NAME=${DATA_NAMES[i]}
  TASK=${DATA_NAMES[i+1]}
  DATA_DIR=${BASE_PATH}/processed_data/${DATA_NAME}/full-1024-512
  for l in ${LR[@]}; do
    for b in ${BS[@]}; do
      for r in ${KD_RATIO[@]}; do
        # Skip successful runs, restart unfinished runs
        directory="${SAVE_PATH}/e${EPOCHS}-bs${b}-lr${l}-G${GRAD_ACC}-N${NPROCS}-NN1-kd${r}"
        if [ -d "$directory/eval" ]; then
          # Count the number of folders in the eval subdirectory
          eval_folder_count=$(find "$directory/eval" -maxdepth 1 -type d | wc -l)
          # Subtract 1 because find includes the parent directory in its count
          let eval_folder_count=eval_folder_count-1
          # Check if there are exactly ${EPOCHS} folders
          if [ "$eval_folder_count" -eq "$EPOCHS" ]; then
              echo "lr${l} - bs${b} - r_kd${r} skipped: ALREADY DONE!!!"
              continue
          fi
          # eval not done yet
          rm -r "${directory}"
          echo "lr${l} - bs${b} - r_kd${r} unfinished: REMOVE AND RERUN!!! (dir removed: ${directory})"
        elif [ -d "$directory" ]; then
          rm -r "${directory}"
        fi
        echo "echo lr${l} - bs${b} - r_kd${r}: RUNNING"
        bash scripts/llama3.1/seqkd/seqkd_8B_70B.sh --nprocs ${NPROCS} \
            --base_path ${BASE_PATH} --wandb_key ${WANDB_KEY} --wandb_prj ${WANDB_PRJ} --model_path ${MODEL_PATH} \
            --model_name ${MODEL_NAME} --teacher_path ${TEACHER_PATH} --teacher_model_name ${TEACHER_MODEL_NAME} \
            --model_type ${MODEL_TYPE} --data_dir ${DATA_DIR} --task ${TASK} --lr ${l} --bs ${b} --kd_ratio ${r} \
            --eval_bs ${EVAL_BS} --epochs ${EPOCHS} --grad_acc ${GRAD_ACC} --max_length ${MAX_LENGTH} \
            --max_prompt_length ${MAX_PROMPT_LENGTH} --save_path ${SAVE_PATH} --save_interval ${SAVE_INTERVAL} \
            --seed ${SEED} --seed_order ${SEED_ORDER} --teacher_peft_path ${TEACHER_LORA_PATH} \
            --teacher_quantized ${TEACHER_QUANTIZED}
      done
    done
  done
done