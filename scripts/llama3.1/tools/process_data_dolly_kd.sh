BASE_PATH="/home/zihaoh/repos/i-am-sober" # path to i-am-sober folder
# model
MODEL_PATH="/home/shared/transformers_cache/hub/<llama-70B/8B-snapshots>" # path to model snapshots
MODEL_TYPE="llama3.1"
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# data
DATA_DIR=${BASE_PATH}/processed_data/dolly/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}
TRAIN_SIZE=15000
DEV_NUM=1000

export TF_CPP_MIN_LOG_LEVEL=3

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --model-path ${MODEL_PATH} \
    --model-type ${MODEL_TYPE} \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --processed-data-dir ${DATA_DIR} \
    --data-process-workers 32 \
    --dev-num ${DEV_NUM}
