# Instructions for Setting Up and Running the Project

## Steps to Prepare and Run the Project

1. **Compile the `.def` file**  
   Compile `pytorch_1.12.1_py310_cuda11.3_cudnn8.def` to create a `.sif` file.  
   **Note:** The `.sif` file is too large to be included in the GitHub repository. Make sure to exclude it from commits by adding it to `.gitignore`.

2. **Modify the `--nodelist` Option in `0_run.sh`**  
   Update the `--nodelist=dgx-a100-n4` argument in `0_run.sh` to match the current node name.

3. **Update the Project Directory in `0_run.sh`**  
   Replace `/scratch/sundong/sjkim/IntentionLearning` in `0_run.sh` with the actual project directory path on your system.

4. **Set the Correct Path for the `.sif` File in `0_run.sh`**  
   Update the `.sif` file location in `0_run.sh` to point to its actual location on your system.

---

## Example Command for Running `0_run.sh`

After making the necessary modifications, you can run the script with the following command:

```bash
./0_run.sh

