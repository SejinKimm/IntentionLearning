# SOLAR (Synthesized Offline Learning data for Abstraction and Reasoning) Generator

SOLAR Generator generates trace data for [Abstraction and Reasoning Corpus(ARC)](https://github.com/fchollet/ARC) tasks to be used for offline reinforcement learning method.

By making it compatible with [ARCLE](https://github.com/ConfeitoHS/arcle) environment, the progress of the trace could be executed and visualized in an ARCLE environment.

| ![gen](./example_image/generate_trace.gif) | ![visgif](./example_image/visualization_example.gif) |
| ------------------------------------------ | ---------------------------------------------------- |

## Setup

### Install SOLAR Generator

SOLAR Generator can be installed by cloning the repository as follows:

```
git clone https://github.com/GIST-DSLab/SOLAR-Generator.git
cd SOLAR-Generator
pip install --upgrade pip
pip install -r requirements.txt

# The following is for creating GIF visualizations on Linux OS.
sudo apt install ffmpeg
```

## File structure

The files are organized according to the following structure.

```
.
├── generate_trace.py
├── generate_trace.sh
├── utils.py
├── visualize_trace.py
├── visualize_trace.sh
├── utils_visualize.py
├── maker
│   ├── base_grid_maker.py
│   ├── grid_maker_example.py
│   ├── ${task_id}
│   │   └── grid_maker.py
│   └── ...
├── ARC_data
│   ├── segment
│   │   ├── ${task_id}
│   │   │   ├──${task_id}_${trace_id}
│   │   │   │   ├──${task_id}_${trace_id}_${segment_id}.json
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── whole
│   │   ├── ${task_id}
│   │   │   ├──${task_id}_${trace_id}.json
│   │   │   └── ...
│   │   └── ...
│   └── wrong
│       ├── ${task_id}
│       │   ├──${task_id}_${trace_id}.json
│       │   └── ...
|       └── ...
├── figure
│   ├── ${task_id}
│   │   ├── gif
│   │   │   ├── pngs_${task_id}_${trace_id}
│   │   │   │   ├── ${i-th step}.png
│   │   │   │   └── ...
│   │   │   ├── ${task_id}_${trace_id}.gif
│   │   │   └── ...
│   │   ├── segment
│   │   │   ├── ${task_id}_${trace_id}_${segment_id}.png
│   │   │   └── ...
│   │   ├── whole
│   │   │   ├── ${task_id}_${trace_id}.png
│   │   │   └── ...
│   │   ├── wrong
│   │   │   ├── ${task_id}_${trace_id}.png
│   │   │   └── ...
│   │   └── inout
│   │       ├── ${task_id}_${trace_id}.png
│   │       └── ...
│   └── ...
├── make_prettier.sh
├── requirement.txt
└── README.md
```

`generate_trace.py` and `generate_trace.sh` files are the main files to generate trace data.

`utils.py` is a collection of utility functions for generating trace

`visualize_trace.py` and `visualize_trace.sh` file are the main files to visualize the trace

`utils_visualize.py` is a collection of utility functions for visualizing trace

`maker` folder contains `base_grid_maker` and `grid_maker_example.py`. `grid_maker.py` exists for each task folder, and is for making new grids that match the conditions in the task. `grid_maker_example.py` base_grid_maker` is for changing data type to handle our data in ARCLE environment.

Created traces are saved in the `ARC_data` folder. Whole traces would be saved in the `whole` folder and partially cropped traces for the specific purpose would be saved in the `segment` folder. If there's any problem in making a trace or the generated output is not correct, the trace will be saved in the `wrong` folder.

When you execute `visualize_trace.py`, the figure will be saved in the `figure` folder.
`gif` for trace gif, `whole` for whole trace, `segment` for segment trace, `wrong` for wrong trace, and `inout` for visualizing only input-output pairs.

`make_prettier.sh` applies prettier to JSON file to make it easy to see. The code to install prettier is below.

```
npm install --save-dev --save-exact prettier
```

## How to use

1. Make a task folder in the `maker` folder.
2. Make `grid_maker.py` inside the task folder and implement it appropriately.
3. Run `generate_trace.py` or `generate_trace.sh` with options.
4. If you want to visualize a trace, run `visualize_trace.py` or `visualize_trace.sh` with options.
5. For more details, each file contains detailed explanations.

### Possible options `generate_trace.py` and `generate_trace.sh`

| Argument               | Description                                  | comment                                                                                    |
| ---------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `env`                  | environment name, version of ARCLE           | `default`: `ARCLE/O2ARCv2Env-v0`                                                           |
| `tasks`                | task_ids for make new trace                  | `required`, can use multiple args, `tasks.txt` file contains task_ids, `all` for all tasks |
| `data_folder_path`     | folder to save trace data                    | `default`: `SOLAR_data` folder <br> if not exist, create folder                            |
| `num_samples`          | number of samples to make in each task       | `default`: 10000                                                                           |
| `num_examples`         | number of example pairs for each trace data  | `default`: 3                                                                               |
| `max_grid_size`        | maximum grid size h, w                       | `default`: (h,w)=(30,30)                                                                   |
| `horizon`              | step length of segment trace                 | `default`: 5                                                                               |
| `save_whole_trace`     | whether save whole trace or not              | `default`: `True`                                                                          |
| `save_seg_trace`       | whether save segment trace or not            | `default`: `True`                                                                          |
| `render_mode`          | whether see the rendering process or not     | `default`: `None` <br> `ansi` to see the rendering                                         |
| `delete_existing_data` | whether delete existing trace or raise error | `default`: `False`, raising an error if data already exists                                |
| `rand_seed`            | random seed that all grid_makers share       | `default`: 0                                                                               |

#### Example code

```
# example for adjusting arguments
python generate_trace.py --tasks 46442a0e --save_whole_trace False --save_seg_trace False --num_samples 100 --max_grid_size 30 30 --horizon 5 --render_mode ansi --delete_existing_data True
```

or you can use a shell file(recommended!)

It is a more efficient method to adjust multiple arguments.

```
sh generate_trace.sh
```

## Image visualization of trace

| Argument           | Description                        | comment                                                             |
| ------------------ | ---------------------------------- | ------------------------------------------------------------------- |
| `mode`             | environment name, version of ARCLE | `default`:`whole` <br>candidate: `segment`, `inout`, `wrong`, `gif` |
| `data_folder_path` | folder path that JSON file exist   | `default`: `SOLAR_data` folder                                      |
| `save_folder_path` | folder path to save figure         | `default`: `figure` folder                                          |
| `file_name`        | JSON file name                     | `required`                                                          |

#### Example code

```
#for gif
python visualize_trace.py --file_name <json_file_name> --mode gif
```

or you can use a shell file(recommended!)

It is a more efficient method to adjust multiple arguments.

```
sh visualize_trace.sh
```

#### Whole trace

![vis](./example_image/visualization_example.png)
