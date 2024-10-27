BASE_PATH="/home/zihaoh/repos/i-am-sober"
ACCELERATE_CONFIG="/home/zihaoh/repos/i-am-sober/configs/accelerate.yml"

# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full-1024-512/llama3.1/"

# model
MODEL_PATH="/home/shared/transformers_cache/hub/<llama-70B-snapshots>"

# runtime
SAVE_PATH="${BASE_PATH}/results/llama3.1/train/sft"

# hp
BS=4
EVAL_BS=8

# wandb
export WANDB_API_KEY="8b07b9ebb0f0b08e31878929ec6324fdc098f376"
export WANDB_PROJECT="i_am_sober_dolly_sft_llama3.1"
export WANDB_NAME="sft-teacher-llama3.1-70B-lora-accelerate"


accelerate launch --config_file $ACCELERATE_CONFIG ${BASE_PATH}/finetune_v1.py --base-path $BASE_PATH --model-path $MODEL_PATH --ckpt-name llama-70B-teacher --model-type llama3.1 --gradient-checkpointing --data-dir $DATA_DIR --num-workers 1 --dev-num -1 --lr 5e-06 --batch-size $BS --eval-batch-size $EVAL_BS --gradient-accumulation-steps 1 --warmup-iters 0 --lr-decay-style cosine --weight-decay 1e-2 --clip-grad 1.0 --epochs 3 --max-length 1024 --max-prompt-length 512 --do-train --do-valid --eval-gen --save-interval -1 --eval-interval -1 --log-interval 4 --mid-log-num -1 --save $SAVE_PATH --seed 10 --seed-order 10 --type lm --do-sample --top-k 0 --top-p 1.0 --temperature 1.0 --peft lora
