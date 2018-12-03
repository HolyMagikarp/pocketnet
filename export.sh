#!/bin/bash
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=../model/model.config
TRAINED_CKPT_PREFIX=../model/model.ckpt-541
mkdir -p object_detection/exported_model
EXPORT_DIR=object_detection/exported_model
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}

sleep 10
