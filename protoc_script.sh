#!/bin/bash
for x in models/research/object_detection/protos/*.proto; do
    protoc ./$x --python_out=.
    echo $x
done

sleep 10
