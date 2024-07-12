BASE_PATH=${1-"/home1/hieutn/cs566/i-am-sober"}
MODEL_PATH=${2-"/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"}
MODEL_TYPE=${3-"llama"}
export TF_CPP_MIN_LOG_LEVEL=3

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_cnn_dailymail.py \
    --hugg-data-id abisee/cnn_dailymail \
    --hugg-data-subset 1.0.0 \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn_dailymail/full \
    --model-path ${MODEL_PATH} \
    --data-process-workers 32 \
    --max-prompt-length 3500 \
    --max-length 4096 \
    --dev-num 1000 \
    --model-type ${MODEL_TYPE}
