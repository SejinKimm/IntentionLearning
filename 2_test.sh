#!/bin/sh

TASK_NAME=$1
MODEL_NAME=$2
TEST_DATASET_NUM=$3
GPU_ID=$4

if [ -z "$TASK_NAME" ]; then
    TASK_NAME=dflip
fi

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=default
fi

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

if [ -z "$TEST_DATASET_NUM" ]; then
    TEST_DATASET_NUM=0
fi

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

echo "Using task: $TASK_NAME" 
echo "Using model: $MODEL_NAME"
echo "Using GPU: $GPU_ID"
echo "Using test data folder: test_$TEST_DATASET_NUM"

if [ "$TEST_DATASET_NUM" -eq 0 ]; then
    for i in 1 2 3 4; do
        python test.py \
            --seed 123 \
            --context_length 5 \
            --batch_size 1 \
            --max_timestep 200 \
            --ckpt_path "./model/" \
            --data_dir_prefix "./dataset/${TASK_NAME}/" \
            --task_name $TASK_NAME \
            --model_name $MODEL_NAME \
            --test_dataset_num $i
    done
else
    python test.py \
        --seed 123 \
        --context_length 5 \
        --batch_size 1 \
        --max_timestep 200 \
        --ckpt_path "./model/${TASK_NAME}/" \
        --data_dir_prefix "./dataset/${TASK_NAME}/" \
        --task_name $TASK_NAME \
        --model_name $MODEL_NAME \
        --test_dataset_num $TEST_DATASET_NUM
fi