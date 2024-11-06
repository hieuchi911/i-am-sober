BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/prompt \
    --model-path ${BASE_PATH}/checkpoints/llama-7B \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type llama

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --model-path ${BASE_PATH}/checkpoints/llama-7B \
    --model-type llama \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --max-prompt-length 256 \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --data-process-workers 32 \
    --dev-num 1000
