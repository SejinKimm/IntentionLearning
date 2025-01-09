#!/bin/bash

srun --nodelist=hpc-pr-a-pod05 --nodes=1 --gres=gpu:1 --exclusive --pty singularity shell --nv --fakeroot -H /scratch/sundong/sjkim pytorch_1.12.1_py310_cuda11.3_cudnn8.sif
