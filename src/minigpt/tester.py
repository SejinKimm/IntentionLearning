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
import logging
import os
import operator
import json
import numexpr as ne

from .PnP import *
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from pathlib import Path
from functools import reduce 

ne.set_num_threads(64)

logger = logging.getLogger(__name__)


class TesterConfig:
    batch_size = 1
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)



class Test_StateActionReturnDataset(Dataset): 

    def __init__(self, data_path, data_type='test_0', step_gap=5, block_size=5*5+2, action_dic={"start":0, "end":1}): 
        self.action_dic = action_dic
        self.vocab_size = len(self.action_dic)
        self.block_size = block_size
        self.step_gap = step_gap
        self.data_path = Path(data_path)
        self.testing_path = self.data_path / data_type
        self.testing_tasks = sorted(os.listdir(self.testing_path))
                
    def __len__(self):
        return len(self.testing_tasks)

    def __getitem__(self, idx):
        task_file = str(self.testing_path / self.testing_tasks[idx])
            
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        total_step = len(task["action_sequence"]["action_sequence"])
        
        states = []
        actions = []
        rtgs = []
        timesteps = []
        pnp = []
        intentions = []
        
        gt_states = []
        gt_actions = []
        gt_rtgs = []
        gt_timesteps = []

        _, _, pnp_init = get_object(task["action_sequence"]["action_sequence"][0]["grid"])
        pnp_init = reduce(operator.add, pnp_init)
        
        for _ in range(self.step_gap-1):
            pnp.append(pnp_init)
            states.append(reduce(operator.add, task["action_sequence"]["action_sequence"][0]["grid"]))
            actions.append(self.action_dic["none"])
            rtgs.append(float(0))
            timesteps.append(0)
            intentions.append([0, 0])

        pnp.append(pnp_init)
        states.append(reduce(operator.add, task["action_sequence"]["action_sequence"][0]["grid"]))
        actions.append(self.action_dic["start"])
        rtgs.append(float(0))
        timesteps.append(0)
        intentions.append([0, 1])
        
        del pnp_init
        
        gt_states.append(reduce(operator.add, task["action_sequence"]["action_sequence"][-1]["grid"]))
        gt_actions.append(self.action_dic["end"])
        gt_rtgs.append(float(1))
        gt_timesteps.append(total_step-1)

        pnp = torch.FloatTensor(pnp)
        states = torch.LongTensor(states)
        states = states.view([-1])
        states = states.contiguous()
        actions = torch.LongTensor(actions)
        rtgs = torch.FloatTensor(rtgs)
        timesteps = torch.LongTensor(timesteps)
        intentions = torch.LongTensor(intentions)

        gt_states = torch.LongTensor(gt_states)
        gt_states = gt_states.view([-1])
        gt_states = gt_states.contiguous()
        gt_actions = torch.LongTensor(gt_actions)
        gt_rtgs = torch.FloatTensor(gt_rtgs)
        gt_timesteps = torch.LongTensor(gt_timesteps)

        return states, actions, rtgs, timesteps, pnp, intentions, gt_states, gt_actions, gt_rtgs, gt_timesteps


class Tester:

    def __init__(self, model, test_dataset, config, args):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config
        self.args = args
        self.num_pixel = args.grid_x * args.grid_y
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = (self.model).to(self.device)

    def test(self):
        model, config = self.model, self.config
        model.eval()

        data = self.test_dataset
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=1,
                            num_workers=config.num_workers)

        correct = 0.0
        too_long = 0
        num_data = len(loader)
        pbar = tqdm(enumerate(loader), total=num_data)
        for it, (x, y, r, t, p, i, gt_x, gt_y, gt_r, gt_t) in pbar:
            
            x = x.to(self.device)
            y = y.to(self.device)
            r = r.to(self.device)
            t = t.to(self.device)
            p = p.to(self.device)
            i = i.to(self.device)
            
            gt_x = gt_x.to(self.device)
            gt_y = gt_y.to(self.device)
            gt_r = gt_r.to(self.device)
            gt_t = gt_t.to(self.device)

            again = True
            step_number = 0
            while(again):
                with torch.no_grad():
                    logits, logits_a, logits_r, logits_src, logits_dst, _, _, _, _, _ = model(states=x, actions=y, targets=False, rtgs=r, timesteps=t, pnp=p, intentions=i) # states, actions, targets, rtgs, timesteps
                step_number = step_number + 1
                
                if(step_number>30):
                    again = False
                    too_long = too_long + 1
                
                elif(torch.argmax(logits_a[0][-1], dim=-1) == self.config.action_dic["end"]):
                    if(torch.equal(x[:, -self.num_pixel:], gt_x)):
                        correct = correct + 1
                        again = False
                    else:
                        again = False
                else:
                    temp = x.clone().detach()
                    x[:, :-self.num_pixel] = temp[:, self.num_pixel:]
                    x[:, -self.num_pixel:] = torch.argmax(logits[:, -self.num_pixel:], dim=-1)
                    
                    if self.config.use_pnp:
                        _, _, pnp_result = get_object((x[:, -self.num_pixel:]).clone().detach().cpu().numpy())
                        pnp_result = reduce(operator.add, pnp_result)
                        pnp_result = torch.FloatTensor(pnp_result).to(self.device)
                        t_p = p.clone().detach()
                        p[:, :-1] = t_p[:, 1:]
                        p[:, -1:] = pnp_result
                        del logits, temp, pnp_result, t_p

                    if self.config.use_intention:
                        temp = i.clone().detach()
                        i[:, :-1, 0] = temp[:, 1:, 0]
                        i[:, -1:, 0] = torch.argmax(logits_src[:, -1:], dim=-1)
                        i[:, :-1, 1] = temp[:, 1:, 1]
                        i[:, -1:, 1] = torch.argmax(logits_dst[:, -1:], dim=-1)
                        del logits_src, logits_dst, temp

                    temp = y.clone().detach()
                    y[:, :-1] = temp[:, 1:]
                    y[:, -1:] = torch.argmax(logits_a[:, -1:], dim=-1)
                    del logits_a, temp
                    
                    temp = r.clone().detach()
                    r[:, :-1] = temp[:, 1:]
                    r[:, -1:] = logits_r[:, -1:]
                    del logits_r, temp
                    
                    temp = t.clone().detach().cpu().numpy()
                    temp_2 = temp.copy()
                    temp[:, :-1] = temp_2[:, 1:]
                    temp[:, -1:] = temp_2[:, -1:] + 1
                    t = torch.LongTensor(temp).to(self.device)
                    del temp, temp_2
                
            pbar.set_description(f"Eval: iter {it+1}/{num_data}, acc: {(100.0*(correct)/(it+1)):.2f}%, too long: {too_long}/{num_data}")
            if(it==num_data-1):
                break
              
        acc = 100.0*float(correct)/num_data
        logger.info("test Acc: %f", acc)
        
        model.train()
        return acc, too_long