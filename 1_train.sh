#!/bin/sh

TASK_NAME=$1
MODEL_NAME=$2
GPU_ID=$3

if [ -z "$TASK_NAME" ]; then
    TASK_NAME=dflip
fi

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=default
fi

if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

echo "Using task: $TASK_NAME"
echo "Using model: $MODEL_NAME"
echo "Using GPU: $GPU_ID"

python src/train.py \
    --seed 123 \
    --context_length 6 \
    --epochs 400 \
    --learning_rate 5e-4 \
    --batch_size 192 \
    --max_timestep 200 \
    --data_dir_prefix "../dataset/${TASK_NAME}/" \
    --train_data_folder 'train' \
    --ckpt_path "../model/${TASK_NAME}/" \
    --save_cycle 20 \
    --n_embd 256 \
    --n_layer 16 \
    --n_head 8  \
    --grid_x 5 \
    --grid_y 5 \
    --intention_size 8 \
    --task_name "$TASK_NAME" \
    --model_name "$MODEL_NAME"

cp "./model/${TASK_NAME}/${TASK_NAME}_${MODEL_NAME}.pt ./model/${TASK_NAME}_${MODEL_NAME}.pt"

export CUDA_VISIBLE_DEVICES=0
