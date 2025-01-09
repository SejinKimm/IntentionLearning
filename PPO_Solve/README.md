# Using PPO to solve ARC Problem
Train ARC Tasks (number: 150, 179, 241, 380) with PPO ([Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)) agent.

# Instructions

## Environments

1. Create a new environment

```bash
conda create --name your_env_name python=3.9
```

2. Activate the environment:
```bash
conda activate your_env_name
```

3. Install pacakges

```bash
pip install -r requirements.txt
```

## How to run

To run the example code (train task 150, eval 150)

```bash
python3 run.py train.task=150 eval.task=150
```

Choose the task within 150, 179, 241, 380

150 - 3 x 3 Horizontal flip task

179 - N x N diagonal flip task

241 - 3 x 3 diagonal flip task

380 - 3 x 3 CCW rotate task

![image](https://github.com/user-attachments/assets/138611b3-824f-47e2-a5ab-35f4362bb960)


# Acknowledge

This implementation is based on the work found at https://github.com/ku-dmlab/arc_trajectory_generator.