#!/bin/bash
NPROCS=2 # number of GPUs to use
MODEL_PARALLEL_SIZE=2
BASE_PATH="/home/zihaoh/repos/i-am-sober" # path to i-am-sober folder
WANDB_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
WANDB_PRJ="i_am_sober_sft13B"
# model
MODEL_PATH="/home/shared/transformers_cache/hub/<llama-13B-snapshots>" # path to model snapshots
MODEL_NAME="llama-13B-teacher"
MODEL_TYPE="llama2"
# hp
LR=(1e-05 5e-06)
BS=(8)
EVAL_BS=8
EPOCHS=3
GRAD_ACC=1
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATASETS=(dolly)
TASK="summ"
# runtime
SAVE_PATH="${BASE_PATH}/results/${MODEL_TYPE}/train/sft"
SAVE_INTERVAL=-1
# seed
SEED=10
SEED_ORDER=10

# run scripts/llama/sft/sft_8B_mp4.sh with all combinations of hyperparameters
for dataset in ${DATASETS[@]}; do
  DATA_DIR=${BASE_PATH}/processed_data/${dataset}/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
  for l in ${LR[@]}; do
    for b in ${BS[@]}; do
      # Skip successful runs, restart unfinished runs
      directory="${SAVE_PATH}/e${EPOCHS}-bs${b}-lr${l}-G${GRAD_ACC}-N${NPROCS}-NN1-mp${MODEL_PARALLEL_SIZE}"
      if [ -d "$directory/eval" ]; then
        # Count the number of folders in the eval subdirectory
        eval_folder_count=$(find "$directory/eval" -maxdepth 1 -type d | wc -l)
        # Subtract 1 because find includes the parent directory in its count
        let eval_folder_count=eval_folder_count-1
        # Check if there are exactly ${EPOCHS} folders
        if [ "$eval_folder_count" -eq "$EPOCHS" ]; then
            echo "lr${l} - bs${b} skipped: ALREADY DONE!!!"
            continue
        fi
        # eval not done yet
        rm -r "${directory}"
        echo "lr${l} - bs${b} unfinished: REMOVE AND RERUN!!! (dir removed: ${directory})"
      elif [ -d "$directory" ]; then
          rm -r "${directory}"
      fi
      echo "echo lr${l} - bs${b}: RUNNING"
      bash scripts/llama/sft/sft_13B_mp4.sh --nprocs ${NPROCS} --model_parallel_size ${MODEL_PARALLEL_SIZE} \
            --base_path ${BASE_PATH} --wandb_key ${WANDB_KEY} --wandb_prj ${WANDB_PRJ} --model_path ${MODEL_PATH} \
            --model_name ${MODEL_NAME} --model_type ${MODEL_TYPE} --data_dir ${DATA_DIR} --task ${TASK} --lr ${l} \
            --bs ${b} --eval_bs ${EVAL_BS} --epochs ${EPOCHS} --grad_acc ${GRAD_ACC} --max_length ${MAX_LENGTH} \
            --max_prompt_length ${MAX_PROMPT_LENGTH} --save_path ${SAVE_PATH} --save_interval ${SAVE_INTERVAL} \
            --seed ${SEED} --seed_order ${SEED_ORDER}
    done
  done
done