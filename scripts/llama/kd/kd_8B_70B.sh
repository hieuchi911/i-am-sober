#!/bin/bash
BASE_PATH=${1-"/home1/hieutn/cs566/i-am-sober"} # path to i-am-sober folder
MODEL_PATH=${2-"/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"}  # path to model snapshots
TEACHER_PATH=${3-"/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"}    # path to model snapshots

# Tokenize data and save in binary files
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_cnn_dailymail.py \
    --hugg-data-id abisee/cnn_dailymail \
    --hugg-data-subset 1.0.0 \
    --processed-data-dir ${BASE_PATH}/processed_data/cnn_dailymail/full \
    --model-path ${MODEL_PATH} \
    --data-process-workers 32 \
    --max-prompt-length 3500 \
    --max-length 4096 \
    --dev-num 1000 \
    --model-type llama

NPROCS=${4-2} # number of GPUs to use
NNODES=1
MODEL_PARALLEL_SIZE=4
BS=8

# Change Model Parallel Size
python tools/convert_mp.py --input_path ${MODEL_PATH} --source_mp_size 1 --target_mp_size ${NPROCS} --model_type llama --exist_ok
python tools/convert_mp.py --input_path ${TEACHER_PATH} --source_mp_size 1 --target_mp_size ${NPROCS} --model_type llama --exist_ok

DISTRIBUTED_ARGS="--nproc_per_node ${NPROCS} \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 2012"

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --teacher-model-path ${TEACHER_PATH}"
OPTS+=" --ckpt-name llama-8B-Student"
OPTS+=" --teacher-ckpt-name llama-70B-Teacher"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${NPROCS}"
OPTS+=" --model-type llama"
OPTS+=" --gradient-checkpointing"
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MODEL_PARALLEL_SIZE}"

# data
OPTS+=" --data-dir ${BASE_PATH}/processed_data/cnn_dailymail/full/llama/"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"

# hp
OPTS+=" --lr 0.00001"
OPTS+=" --batch-size ${BS}"
OPTS+=" --eval-batch-size ${BS}"
OPTS+=" --gradient-accumulation-steps 1"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 3"
OPTS+=" --kd-ratio 0.5"

# length
OPTS+=" --max-length 4096"
OPTS+=" --max-prompt-length 3500"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save results/llama/train/kd"

# seed
OPTS+=" --seed 10"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# type
OPTS+=" --type kd"

# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
${CMD}
