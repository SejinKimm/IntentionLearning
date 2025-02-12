#!/bin/sh

TASK_NAME=$1
TRAIN_OR_TEST=$2

if [ -z "$TASK_NAME" ]; then
    TASK_NAME=dflip
fi

if [ -z "$TRAIN_OR_TEST" ]; then
    TRAIN_OR_TEST=train
fi

python src/intention_preprocess.py "./dataset/$TASK_NAME/$TRAIN_OR_TEST"
