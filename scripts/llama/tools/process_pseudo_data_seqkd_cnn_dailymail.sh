export TF_CPP_MIN_LOG_LEVEL=3

BASE_PATH="/home/zihaoh/repos/i-am-sober" # path to i-am-sober folder
# model
MODEL_PATH=""  # path to model snapshots
MODEL_NAME="llama-13B-sft"
MODEL_TYPE="llama2"
# length
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
# dataset
DEV_NUM=-1
DATA_DIR=${BASE_PATH}/results/${MODEL_TYPE}/gen/${MODEL_NAME}/t1.0-l${MAX_LENGTH}
PROCESSED_DATA_DIR=${BASE_PATH}/processed_data/cnn_dailymail/pseudo

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${DATA_DIR} \
    --processed-data-dir ${PROCESSED_DATA_DIR} \
    --model-path ${MODEL_PATH}\
    --model-type ${MODEL_TYPE} \
    --data-process-workers 32 \
    --max-length ${MAX_LENGTH} \
    --max-prompt-length ${MAX_PROMPT_LENGTH} \
    --dev-num ${DEV_NUM}

cp ${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}/${MODEL_TYPE}/valid_0.bin ${BASE_PATH}/processed_data/cnn_dailymail/pseudo/${MODEL_TYPE}/
cp ${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}/${MODEL_TYPE}/valid_0.idx ${BASE_PATH}/processed_data/cnn_dailymail/pseudo/${MODEL_TYPE}/
cp ${BASE_PATH}/processed_data/cnn_dailymail/full-${MAX_LENGTH}-${MAX_PROMPT_LENGTH}/${MODEL_TYPE}/valid.jsonl ${BASE_PATH}/processed_data/cnn_dailymail/pseudo/${MODEL_TYPE}/
