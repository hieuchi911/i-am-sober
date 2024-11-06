#!/bin/bash

# model type
MODEL_TYPE=llama3.1
# triples of (source_mp_size   target_mp_size   absolute_paths)
MODEL_PATHS=(
# 1 4 /home/shared/transformers_cache/hub/<llama-70B-snapshots>
# 1 4 /home/shared/transformers_cache/hub/<llama-8B-snapshots>
1 4 /scratch1/hieutn/hub/models--meta-llama--Meta-Llama
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
    
    python tools/convert_mp.py ${ARGS}
done
