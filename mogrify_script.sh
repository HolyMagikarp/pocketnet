#!/bin/bash
cd dataset
for x in */; do
    echo $x
    mogrify -strip $x/*
done

sleep 10
