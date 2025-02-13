"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import torch
import torch.optim as optim
import wandb
import logging
import subprocess
import os
import operator
import json

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from functools import reduce 

logger = logging.getLogger(__name__)


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-2
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 
    ckpt_path = None
    num_workers = 0 

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Train_StateActionReturnDataset(Dataset):
    
    def __init__(self, data_path, data_type=None, step_gap=5, action_dic={"start":0, "end":1}): 
        self.action_dic = action_dic
        self.vocab_size = len(self.action_dic)
        self.step_gap = step_gap
        self.data_path = Path(data_path)
        self.training_path = self.data_path / data_type
        self.training_tasks = []

        for path, _, files in os.walk(self.training_path):
            for name in files:
                self.training_tasks.append(os.path.join(path, name))

        self.num_dataset = len(self.training_tasks)
        
    def __len__(self):
        return self.num_dataset

    def __getitem__(self, idx):
        task_file = self.training_tasks[idx]
            
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        states = []
        actions_str = []
        actions = []
        rtgs = []
        timesteps = []
        pnp = []
        intentions = []
                
        for i in range(len(task["action_sequence"]["action_sequence"])):
            pnp.append(reduce(operator.add, task["action_sequence"]["action_sequence"][i]["pnp"]))
            states.append(reduce(operator.add, task["action_sequence"]["action_sequence"][i]["grid"]))
            actions_str.append(task["action_sequence"]["action_sequence"][i]["action"]["tool"])
            actions.append(self.action_dic[task["action_sequence"]["action_sequence"][i]["action"]["tool"]])
            rtgs.append(float(task["action_sequence"]["action_sequence"][i]["reward"]))
            timesteps.append(int(task["action_sequence"]["action_sequence"][i]["time_step"]))

            if "intention" in task["action_sequence"]["action_sequence"][i]:
                intentions.append(task["action_sequence"]["action_sequence"][i]["intention"])
            else:
                intentions.append([0, 0])
        
        pnp = torch.LongTensor(pnp)
        states = torch.LongTensor(states)
        states = states.view([-1])
        states = states.contiguous()
        actions = torch.LongTensor(actions)
        rtgs = torch.FloatTensor(rtgs)
        timesteps = torch.LongTensor(timesteps)
        intentions = torch.LongTensor(intentions)

        return states, actions, rtgs, timesteps, pnp, intentions


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = (self.model).to(self.device)

        wandb.init(project="ARC_Decision_Transformer", config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs
        })

    def save_checkpoint(self):
        logger.info("saving %s", self.config.ckpt_path)
        model_name = self.config.task_name + "_" + self.config.model_name + ".pt"

        model_path = self.config.ckpt_path + "/" + model_name

        torch.save(self.model, model_path)
        wandb.save(model_path)

        return model_name
        
    def train(self):
        model, config = self.model, self.config
        optimizer = model.configure_optimizers(config)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75)
        
        def run_epoch(split, epoch_num=0, total_epoch=None):
            is_train = split == 'train'
            model.train(is_train)

            train_data = self.train_dataset
            loader = DataLoader(train_data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            epoch_loss = 0

            for it, batch in pbar:
                x, a, r, t = batch[:4]

                x = x.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                p = batch[4].to(self.device) if self.config.use_pnp else None
                i = batch[5].to(self.device) if self.config.use_intention else None
                
                with torch.set_grad_enabled(is_train):
                    if self.config.use_pnp and self.config.use_intention:
                        _, _, _, _, _, loss, loss_s, loss_a, loss_r, loss_i = model(states=x, actions=a, targets=True, rtgs=r, timesteps=t, pnp=p, intentions=i)
                    elif self.config.use_pnp:
                        _, _, _, _, _, loss, loss_s, loss_a, loss_r, _ = model(states=x, actions=a, targets=True, rtgs=r, timesteps=t, pnp=p)
                    elif self.config.use_intention:
                        _, _, _, _, _, loss, loss_s, loss_a, loss_r, loss_i = model(states=x, actions=a, targets=True, rtgs=r, timesteps=t, intentions=i)
                    else:
                        _, _, _, _, _, loss, loss_s, loss_a, loss_r, _ = model(states=x, actions=a, targets=True, rtgs=r, timesteps=t)

                    epoch_loss += loss.item()
                   

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    pbar.set_description(f"Train| epoch: {epoch+1}/{total_epoch}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.3e}")


                wandb.log({
                    "etc/epoch": epoch_num,
                    "etc/batch": it,
                    "etc/lr": optimizer.param_groups[0]['lr'],
                    "loss/total_loss": loss.item(),
                    "loss/state_loss": loss_s.item(),
                    "loss/action_loss": loss_a.item(),
                    "loss/rtg_loss": loss_r.item(),
                })            

                if self.config.use_intention:
                    wandb.log({"loss/intention_loss": loss_i.item()})
                else:
                    wandb.log({"loss/intention_loss": 0})

            return epoch_loss

        val_acc = 0.
        for epoch in range(config.max_epochs):
            epoch_loss = run_epoch('train', epoch_num=epoch, total_epoch=config.max_epochs)

            scheduler.step()
                    
            if self.config.ckpt_path is not None and epoch%self.config.save_cycle==0:
                self.save_checkpoint()
                test_script = f"./2_test.sh {self.config.task_name} {self.config.model_name} 1 {self.config.gpu_id}"

                process = subprocess.run(test_script, shell=True, capture_output=True, text=True)
                #print(process.stdout)
                print(process.stderr)

                val_acc = float(process.stderr.strip().split()[-1]) if process.stderr.strip() else 0.0
            wandb.log({
                "metrics/epoch": epoch,
                "metrics/epoch_loss": epoch_loss,
                "metrics/val_acc": val_acc
            })

    