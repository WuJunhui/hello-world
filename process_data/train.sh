#!/usr/bin/env sh
LOG="logs/train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
python \
    solve.py 1 2>&1 | tee "$LOG"
