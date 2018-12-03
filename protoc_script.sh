#!/bin/bash
# RUN FROM tensorflow/models/research
for x in object_detection/protos/*.proto; do
    protoc ./$x --python_out=.
done

sleep 10
