#!/bin/bash

CAFFEROOT=/export/home/xxx/caffe-fl-sig
LOG="logs/train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")

echo Logging output to "$LOG"
time $CAFFEROOT/build/tools/caffe train \
    --solver=solver.prototxt \
    --weights=resnext50.caffemodel \
    --gpu=3

