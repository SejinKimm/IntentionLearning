import torch
import logging
import argparse

from minigpt.utils import set_seed
from minigpt.tester import Tester, TesterConfig, Test_StateActionReturnDataset
from minigpt.PnP import *

ACTION_DIC={
    "start":0,
    "undo":1,
    "edit":2,
    "copyFromInput":3,
    "rotate":4,
    "reflecty":5,
    "reflectx":6,
    "translate":7,
    "resizeOutputGrid":8,
    "resetOutputGrid":9,
    "selected_cells":10, 
    "select_fill":11,
    "none":12,
    "end":13,
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=3e-4)

parser.add_argument('--grid_x', type=int, default=5)
parser.add_argument('--grid_y', type=int, default=5)
parser.add_argument('--color_num', type=int, default=10)
parser.add_argument('--max_timestep', type=int, default=200)

parser.add_argument('--data_dir_prefix', type=str, default='./dataset/dflip/')
parser.add_argument('--test_dataset_num', type=str, default=1)
parser.add_argument('--ckpt_path', type=str, default='./model/')
parser.add_argument('--task_name', type=str, default='dflip')
parser.add_argument('--model_name', type=str, default='default')

args = parser.parse_args()


set_seed(args.seed)
logger = logging.getLogger(__name__)
    
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

test_dataset = Test_StateActionReturnDataset(
    data_path=args.data_dir_prefix,
    data_type="test_" + args.test_dataset_num, 
    step_gap=args.context_length, 
    block_size=args.context_length,
    action_dic=ACTION_DIC
)

model_name = args.task_name + "_" + args.model_name + ".pt"
model_path = args.ckpt_path + model_name

test_conf = TesterConfig(
    batch_size=args.batch_size, 
    num_workers=4, 
    seed=args.seed, 
    ckpt_path=model_path,
    action_dic=ACTION_DIC,
    use_pnp = args.model_name in ["pnp", "pnp_intention"],
    use_intention = args.model_name in ["intention", "pnp_intention"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(model_path, map_location=device, weights_only=False)

tester = Tester(model, test_dataset, test_conf, args)

test_acc = tester.test()