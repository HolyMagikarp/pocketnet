#!/bin/bash
# RUN FROM tensorflow/models/research/
# change model.ckpt-### into the checkpoint number you want
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=object_detection/required_export_files/model.config
TRAINED_CKPT_PREFIX=object_detection/required_export_files/model.ckpt-541
EXPORT_DIR=object_detection/exported_model
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}

sleep 10
