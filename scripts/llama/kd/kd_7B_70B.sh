BASE_PATH=${1-"/home1/hieutn/cs566-test/i-am-sober"} # path to i-am-sober folder
MODEL_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"  # path to model snapshots
TEACHER_PATH="/scratch1/hieutn/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/"    # path to model snapshots

NPROCS=2 # number of GPUs to use

# Tokenize data and save in binary files
PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${MODEL_PATH} \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type llama

# Change Model Parallel Size
python tools/convert_mp.py --input_path ${MODEL_PATH} --source_mp_size 1 --target_mp_size ${NPROCS} --model_type llama --exist_ok
python tools/convert_mp.py --input_path ${TEACHER_PATH} --source_mp_size 1 --target_mp_size ${NPROCS} --model_type llama --exist_ok

# Finetune model
torchrun --nproc_per_node ${NPROCS} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 2012 \
finetune.py --base-path ${BASE_PATH} \
--model-path ${MODEL_PATH} \
--teacher-model-path ${TEACHER_PATH} \
--ckpt-name llama-8B-Student \
--teacher-ckpt-name llama-70B-Teacher \
--teacher-model-fp16 \
--n-gpu ${NPROCS} \
--model-type llama \
--gradient-checkpointing \
--model-parallel --model-parallel-size ${NPROCS} \
--data-dir processed_data/dolly/full/llama/ \
--num-workers 4 \
--dev-num 1000 \
--lr 0.00001 \
--batch-size 8 \
--eval-batch-size 8 \
--gradient-accumulation-steps 1 \
--warmup-iters 0 --lr-decay-style cosine --weight-decay 1e-2 --clip-grad 1.0 \
--epochs 10 \
--kd-ratio 0.5 \
--max-length 512 \
--max-prompt-length 256 \
--do-train --do-valid --eval-gen \
--save-interval -1 --eval-interval -1 --log-interval 4 --mid-log-num -1 \
--save results/llama/train/kd \
--seed 10 \
--deepspeed --deepspeed_config configs/deepspeed/ds_config.json \
--type kd \
--do-sample --top-k 0 --top-p 1.0 --temperature 1.0
