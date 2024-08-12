#!/bin/bash

# model type
MODEL_TYPE=llama2
# triples of (source_mp_size   target_mp_size   absolute_paths)
MODEL_PATHS=(
    4  1 /scratch1/hieutn/ckps-selfkd-hpo-new/e3-bs8-lr5e-06-G1-N4-NN1-kd1.0-mp4/3750/
    4  1 /scratch1/hieutn/ckps-sft-hpo/e3-bs4-lr5e-06-G1-N4-NN1-mp4/7500/
)

for ((i=0; i<${#MODEL_PATHS[@]}; i+=3))
do
    SRC_SIZE=${MODEL_PATHS[i]}
    TAR_SIZE=${MODEL_PATHS[i+1]}
    MODEL_PATH=${MODEL_PATHS[i+2]}
    # echo the model path and mp size
    ARGS=""
    ARGS+="--input_path ${MODEL_PATH}"
    ARGS+=" --source_mp_size ${SRC_SIZE}"
    ARGS+=" --target_mp_size ${TAR_SIZE}"
    ARGS+=" --model_type ${MODEL_TYPE}"
    ARGS+=" --exist_ok"
    echo "python tools/convert_mp.py ${ARGS}"
    
    # python tools/convert_mp.py ${ARGS}
done
