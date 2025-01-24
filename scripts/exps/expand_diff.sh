#!/bin/bash

SCALE=7.5

# DATASET="caltech-101"
# STRENGTH=0.5

DATASET="m_rock"
STRENGTH=0.1

# DATASET="rock_minerals"
# STRENGTH=0.3

START=9
PERIOD=5
CON=0.2
K=3
EXPAND_NUM=3
GPU=0
SPLIT=0
GUIDANCE_TYPE="transform_guidance"
RHO=10.0
GUIDE_MODEL="resnet50"
GUIDE_MODEL_WEIGHT="checkpoint/${DATASET}/resnet50_unpretrained_lr0.1/seed1/model_best.pth.tar"

DATA_SAVE_PATH=data/${DATASET}_expansion/save/distdiff_batch_${EXPAND_NUM}x
CUDA_VISIBLE_DEVICES=${GPU} python generate_data.py \
        --guidance_type=${GUIDANCE_TYPE}  -a ${GUIDE_MODEL} -d ${DATASET} \
        --output_dir ${DATA_SAVE_PATH} --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --gradient_checkpointing --K ${K} --train_batch_size 4 --optimize_targets "global_prototype-local_prototype" \
        --strength ${STRENGTH} --num_images_per_prompt ${EXPAND_NUM} --guidance_step ${START} --guidance_period ${PERIOD} \
        --encoder_weight_path ${GUIDE_MODEL_WEIGHT} --guidance_scale ${SCALE} --constraint_value ${CON} --rho ${RHO} --total_split 1 --split ${SPLIT}
