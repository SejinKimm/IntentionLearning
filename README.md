# Instructions for Setting Up and Running the Project

## Steps to Prepare and Run the Project

### 1. **Compile the `.def` file to create a `.sif` file**
The first step is to build a Singularity Image Format (`.sif`) file from the provided definition file (`.def`). There are two methods to do this:

- **Using a Local Singularity Installation**  
  If you have Singularity installed on your system, run:

  ```bash
  singularity build pytorch_1.12.1_py310_cuda11.3_cudnn8.sif pytorch_1.12.1_py310_cuda11.3_cudnn8.def
  ```

- **Using a Remote Cluster with a Temporary Sandbox**  
  If you are working on a shared HPC cluster, you may need to use a writable sandbox first:

  ```bash
  singularity build --sandbox pytorch_container/ pytorch_1.12.1_py310_cuda11.3_cudnn8.def
  singularity build pytorch_1.12.1_py310_cuda11.3_cudnn8.sif pytorch_container/
  ```

**Note:**  
- The `.sif` file is too large to be included in the GitHub repository. **Make sure to exclude it from commits by adding it to `.gitignore`.**  
- The `.sif` file generated in this step will be used later in **Step 3** to run the Singularity shell.

---

### 2. **Modify Resource Allocation Options in `0_run.sh`**
The script `0_run.sh` uses the `srun` command to allocate computing resources. Update the following options in the script:

```bash
# --nodelist: Specify the exact node where the job will run.
# Replace 'dgx-a100-n4' with your actual node name.
--nodelist=dgx-a100-n4

# --nodes: Define the number of nodes required for the job.
# By default, the script uses one node.
--nodes=1

# --gres: Request GPU resources.
# For example, to allocate 2 GPUs instead of 1, modify as follows:
--gres=gpu:1
```

---

### 3. **Update Paths and Run the Singularity Shell**
The `0_run.sh` script launches a Singularity container on the allocated resources. You need to update the following paths:

- **Set the correct project directory (`-H` option)**  
  The `-H` flag in the `singularity shell` command sets the home directory **inside the container** to match a specific directory on the host system.  
  Update the following line in `0_run.sh` to reflect the actual project path:

  ```bash
  -H /scratch/sundong/sjkim/IntentionLearning
  ```

- **Specify the correct `.sif` file path**  
  Ensure that the `.sif` file you compiled in **Step 1** is correctly referenced in `0_run.sh`. Modify the path as necessary:

  ```bash
  ../pytorch_1.12.1_py310_cuda11.3_cudnn8.sif
  ```

Thus, the final `srun` command in `0_run.sh` should look like:

```bash
srun \
  --nodelist=dgx-a100-n4 \
  --nodes=1 \
  --gres=gpu:1 \
  --exclusive \
  --pty \
  singularity shell \
  --nv \
  -H /scratch/sundong/sjkim/IntentionLearning \
  ../pytorch_1.12.1_py310_cuda11.3_cudnn8.sif
```

---

## Example Command for Running `0_run.sh`

After making the necessary modifications, execute the script using:

```bash
./0_run.sh
```

Once executed, you will have access to the **Singularity shell** inside the container.

---

## Notes
- Verify resource availability using cluster management tools (e.g., `sinfo`) before running the script.
- Ensure that the `.sif` file has been successfully compiled in **Step 1** before proceeding to Step 3.
- Avoid committing the `.sif` file to the repository, as it is large and unnecessary for version control.

---
