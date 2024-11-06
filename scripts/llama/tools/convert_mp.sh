#!/bin/bash

# model type
MODEL_TYPE=llama2
# triples of (source_mp_size   target_mp_size   absolute_paths)

# 4  1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4/8574/
# 4  1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4/8574/
# 4  1 /scratch1/hieutn/ckps-selfkd-hpo-new/e3-bs8-lr5e-06-G1-N4-NN1-kd1.0-mp4/3750/
# 4  1 /scratch1/hieutn/ckps-sft-hpo/e3-bs4-lr5e-06-G1-N4-NN1-mp4/7500/

MODEL_PATHS=(
# # 4 1 /scratch1/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd1.0-mp4/8574
# # 4 1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr1e-05-G1-N4-NN1-mp4/8574
# # 4 1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-05-G1-N4-NN1-mp4/8574
# # 4 1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-05-G1-N4-NN1-mp4/4287
# # 4 1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-06-G1-N4-NN1-mp4/4287
# 4 1 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr1e-05-G1-N4-NN1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-05-G1-N4-NN1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-05-G1-N4-NN1-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-06-G1-N4-NN1-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr1e-05-G1-N4-NN1-kd0.1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr1e-05-G1-N4-NN1-kd1.0-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr1e-05-G1-N4-NN1-kd10.0-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-05-G1-N4-NN1-kd0.1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-05-G1-N4-NN1-kd1.0-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-05-G1-N4-NN1-kd10.0-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd0.1-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd1.0-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd10.0-mp4/8574
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr1e-05-G1-N4-NN1-kd0.1-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr1e-05-G1-N4-NN1-kd1.0-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr1e-05-G1-N4-NN1-kd10.0-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-05-G1-N4-NN1-kd0.1-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-05-G1-N4-NN1-kd1.0-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-05-G1-N4-NN1-kd10.0-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-06-G1-N4-NN1-kd1.0-mp4/4287
4 1 /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-06-G1-N4-NN1-kd0.1-mp4/4287
)

for ((i=0; i<${#MODEL_PATHS[@]}; i+=3))
do
    SRC_SIZE=${MODEL_PATHS[i]}
    TAR_SIZE=${MODEL_PATHS[i+1]}
    MODEL_PATH=${MODEL_PATHS[i+2]}
    ls $MODEL_PATH
    # # echo the model path and mp size
    # ARGS=""
    # ARGS+="--input_path ${MODEL_PATH}"
    # ARGS+=" --source_mp_size ${SRC_SIZE}"
    # ARGS+=" --target_mp_size ${TAR_SIZE}"
    # ARGS+=" --model_type ${MODEL_TYPE}"
    # ARGS+=" --exist_ok"
    # echo "python tools/convert_mp.py ${ARGS}"
    
    # python tools/convert_mp.py ${ARGS}
done

# UN-HARNESSED MODELS================================================================================
# # KD models:
# # /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd1.0-mp4

# # SFT baselines:            
# # /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr1e-05-G1-N4-NN1-mp4
# # /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-05-G1-N4-NN1-mp4
# # /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr1e-05-G1-N4-NN1-mp4     # not exist
# # /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-05-G1-N4-NN1-mp4
# # /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-06-G1-N4-NN1-mp4

# models=(
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd1.0-mp4/8574 /scratch1/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd1.0-mp4
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr1e-05-G1-N4-NN1-mp4/8574 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr1e-05-G1-N4-NN1-mp4
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-05-G1-N4-NN1-mp4/8574 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-05-G1-N4-NN1-mp4
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-05-G1-N4-NN1-mp4/4287 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-05-G1-N4-NN1-mp4
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-06-G1-N4-NN1-mp4/4287 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-06-G1-N4-NN1-mp4
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4/8574 /scratch1/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4
# )

# # loop through each pairs of `models` and copy the folder from source to target
# for ((i=0; i<${#models[@]}; i+=2))
# do
#     SRC=${models[i]}
#     TAR=${models[i+1]}
#     mkdir -p ${TAR}
#     echo "cp -r ${SRC} ${TAR}"
#     cp -r ${SRC} ${TAR}
# done

# # VAL SCORES FROM FINETUNE=============================================================================
# # KDs
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr1e-05-G1-N4-NN1-kd0.1-mp4/log.txt      33.496
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr1e-05-G1-N4-NN1-kd1.0-mp4/log.txt      33.3559
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr1e-05-G1-N4-NN1-kd10.0-mp4/log.txt     32.8133
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-05-G1-N4-NN1-kd0.1-mp4/log.txt      29.0731
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-05-G1-N4-NN1-kd1.0-mp4/log.txt      29.3698
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-05-G1-N4-NN1-kd10.0-mp4/log.txt     28.2997
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd0.1-mp4/log.txt      34.1004
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd1.0-mp4/log.txt      34.2624
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs4-lr5e-06-G1-N4-NN1-kd10.0-mp4/log.txt     33.9276
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr1e-05-G1-N4-NN1-kd0.1-mp4/log.txt      33.9831
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr1e-05-G1-N4-NN1-kd1.0-mp4/log.txt      32.9849
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr1e-05-G1-N4-NN1-kd10.0-mp4/log.txt     33.6989
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-05-G1-N4-NN1-kd0.1-mp4/log.txt      31.1235
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-05-G1-N4-NN1-kd1.0-mp4/log.txt      30.4382
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-05-G1-N4-NN1-kd10.0-mp4/log.txt     29.6635
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-06-G1-N4-NN1-kd0.1-mp4/log.txt      33.9797
# /project/lerman_316/hieutn/ckps-kd-dolly-13b-7b/e3-bs8-lr5e-06-G1-N4-NN1-kd1.0-mp4/log.txt      32.8918

# # SFTs
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr1e-05-G1-N4-NN1-mp4/log.txt               35.1824
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-05-G1-N4-NN1-mp4/log.txt               31.0063
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs4-lr5e-06-G1-N4-NN1-mp4/log.txt               35.386
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr1e-05-G1-N4-NN1-mp4/log.txt               -
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-05-G1-N4-NN1-mp4/log.txt               32.8416
# /project/lerman_316/hieutn/ckps-sft-dolly-7b/e3-bs8-lr5e-06-G1-N4-NN1-mp4/log.txt               35.3786