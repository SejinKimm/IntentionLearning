import logging
import argparse

from minigpt.utils import set_seed
from minigpt.DT_ARC import GPT, GPTConfig
from minigpt.trainer import Trainer, TrainerConfig, Train_StateActionReturnDataset



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
parser.add_argument('--context_length', type=int, default=6)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--lr_decay', type=bool, default=False)

parser.add_argument('--intention_size', type=int, default=8)
parser.add_argument('--grid_x', type=int, default=5)
parser.add_argument('--grid_y', type=int, default=5)
parser.add_argument('--color_num', type=int, default=10)
parser.add_argument('--max_timestep', type=int, default=200)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)

parser.add_argument('--data_dir_prefix', type=str, default='./dataset/dflip/')
parser.add_argument('--train_data_folder', type=str, default='train/')
parser.add_argument('--ckpt_path', type=str, default='./model/')
parser.add_argument('--save_cycle', type=int, default=200)
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

train_dataset = Train_StateActionReturnDataset(
    data_path=args.data_dir_prefix,
    data_type=args.train_data_folder, 
    step_gap=args.context_length, 
    action_dic=ACTION_DIC
)

mconf = GPTConfig(
    vocab_size=len(ACTION_DIC), 
    block_size = args.context_length,
    n_layer=args.n_layer, 
    n_head=args.n_head, 
    n_embd=args.n_embd,
    max_timestep=args.max_timestep,
    grid_x=args.grid_x,
    grid_y=args.grid_y,
    color_num=args.color_num,
    context_length=args.context_length,
    embd_pdrop = args.embd_pdrop,
    resid_pdrop = args.resid_pdrop,
    attn_pdrop = args.attn_pdrop,
    intention_size = args.intention_size,
    use_pnp = args.model_name in ["pnp", "pnp_intention"],
    use_intention = args.model_name in ["intention", "pnp_intention"],
)

tconf = TrainerConfig(
    max_epochs=args.epochs, 
    batch_size=args.batch_size, 
    learning_rate=args.learning_rate,
    num_workers=4, 
    seed=args.seed, 
    ckpt_path=args.ckpt_path,
    save_cycle=args.save_cycle,
    task_name=args.task_name,
    model_name=args.model_name,
    use_pnp = args.model_name in ["pnp", "pnp_intention"],
    use_intention = args.model_name in ["intention", "pnp_intention"],
)

model = GPT(mconf)

trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()