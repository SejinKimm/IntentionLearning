#!/bin/bash

srun --nodelist=dgx-a100-n4 --nodes=1 --gres=gpu:1 --exclusive --pty singularity shell --nv -H /scratch/sundong/sjkim/IntentionLearning ../pytorch_1.12.1_py310_cuda11.3_cudnn8.sif
