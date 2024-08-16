#!/bin/bash
NPROCS=2 # number of GPUs to use
MODEL_PARALLEL_SIZE=2
BASE_PATH="/home/zihaoh/repos/i-am-sober" # path to i-am-sober folder
WANDB_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
WANDB_PRJ="i_am_sober"
# model
MODEL_PATH="/home/shared/transformers_cache/hub/<llama-7B-snapshots>" # path to model snapshots
MODEL_NAME="llama-7B-Student"
TEACHER_PATH="" # path to SFT teacher snapshots
TEACHER_MODEL_NAME="llama-13B-Teacher"
MODEL_TYPE="llama2"
# hp
LR=(5e-05 1e-05 5e-06)
BS=(8 4)
KD_RATIO=(0.1 1.0 10.0)
EVAL_BS=8
EPOCHS=3
GRAD_ACC=1
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_NAMES=(
  dolly general
) # (dolly general cnn_dailymail summ)
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/kd"
SAVE_INTERVAL=-1
# seed
SEED=10
SEED_ORDER=10

# run scripts/llama/kd/kd_8B_70B.sh with all combinations of hyperparameters
for ((i=0; i<${#DATA_NAMES[@]}; i+=2))
  DATA_NAME=${DATA_NAMES[i]}
  TASK=${DATA_NAMES[i+1]}
  DATA_DIR=${BASE_PATH}/processed_data/${DATA_NAME}/pseudo
  for l in ${LR[@]}; do
    for b in ${BS[@]}; do
      for r in ${KD_RATIO[@]}; do
        # Skip successful runs, restart unfinished runs
        directory="${SAVE_PATH}/e${EPOCHS}-bs${b}-lr${l}-G${GRAD_ACC}-N${NPROCS}-NN1-kd${r}-mp${MODEL_PARALLEL_SIZE}"
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
        bash scripts/llama/seqkd/seqkd_7B_13B.sh --nprocs ${NPROCS} --model_parallel_size ${MODEL_PARALLEL_SIZE} \
            --base_path ${BASE_PATH} --wandb_key ${WANDB_KEY} --wandb_prj ${WANDB_PRJ} --model_path ${MODEL_PATH} \
            --model_name ${MODEL_NAME} --teacher_path ${TEACHER_PATH} --teacher_model_name ${TEACHER_MODEL_NAME} \
            --model_type ${MODEL_TYPE} --data_dir ${DATA_DIR} --task ${TASK} --lr ${l} --bs ${b} --kd_ratio ${r} \
            --eval_bs ${EVAL_BS} --epochs ${EPOCHS} --grad_acc ${GRAD_ACC} --max_length ${MAX_LENGTH} \
            --max_prompt_length ${MAX_PROMPT_LENGTH} --save_path ${SAVE_PATH} --save_interval ${SAVE_INTERVAL} \
            --seed ${SEED} --seed_order ${SEED_ORDER}
      done
    done
  done
done