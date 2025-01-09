import json
import numpy as np
import os
from typing import List
from numpy.typing import NDArray

from arcle.loaders import ARCLoader
from arcle.loaders import MiniARCLoader
from arcle.loaders import ARCLoader
from gymnasium.core import ObsType, ActType
import numpy as np
import os
import copy
from numpy.typing import NDArray
from typing import List
import json
import pickle
import yaml


class SizeConstrainedLoader(ARCLoader):
    def __init__(self, size, train=True) -> None:
        self.size = size
        super().__init__(train=train)
    
    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                problem = json.load(fp)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []


                for d in problem['train']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    if inp.shape[0] > self.size or inp.shape[1] > self.size or oup.shape[0] > self.size or oup.shape[1] > self.size:
                        continue
                    ti.append(inp)
                    to.append(oup)

                for d in problem['test']:
                    inp = np.array(d['input'],dtype=np.uint8)
                    oup = np.array(d['output'],dtype=np.uint8)
                    if inp.shape[0] > self.size or inp.shape[1] > self.size or oup.shape[0] > self.size or oup.shape[1] > self.size:
                        continue
                    ei.append(inp)
                    eo.append(oup)

                if len(ti) == 0:
                    continue

                desc = {'id': os.path.basename(fp.name).split('.')[0]}
                dat.append((ti,to,ei,eo,desc))
                
        return dat
    
class MiniARCLoader(MiniARCLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, **kwargs):
        
        dat = []

        for p in self._pathlist:
            with open(p) as fp:
                fpdata = fp.read().replace('null', '"0"')
                problem = json.loads(fpdata)

                ti: List[NDArray] = []
                to: List[NDArray] = []
                ei: List[NDArray] = []
                eo: List[NDArray] = []

                for d in problem['train']:
                    ti.append(np.array(d['input'],dtype=np.uint8))
                    to.append(np.array(d['output'],dtype=np.uint8))
                
                for d in problem['test']:
                    ei.append(np.array(d['input'],dtype=np.uint8))
                    eo.append(np.array(d['output'],dtype=np.uint8))

                fns = os.path.basename(fp.name).split('_')
                desc = {'id': fns[-1].split('.')[-2], 'description': ' '.join(fns[0:-1]).strip() }

                dat.append((ti,to,ei,eo,desc))
                
        return dat

class EntireSelectionLoader(ARCLoader):
    def __init__(self, train_task: str, eval_task: str) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, 'dataset')
        self.config_path = os.path.join(current_dir, 'ppo/ppo_config_entsel.yaml')
        config = load_config(self.config_path)
        self.train_task = train_task or config['train']['task']
        self.eval_task = eval_task or config['eval']['task']

        self.train_file = os.path.join(self.data_dir, f'train_{self.train_task}.pkl')
        self.eval_file = os.path.join(self.data_dir, f'eval_{self.eval_task}.pkl')
        super().__init__()

    
    def parse(self, **kwargs):
        dat = []

        ti: List[NDArray] = []
        to: List[NDArray] = []
        ei: List[NDArray] = []
        eo: List[NDArray] = []
        
        print("path:", os.path)
        if not os.path.exists(self.train_file):
            for p in self._pathlist:

                with open(p) as fp:
                    problem = json.load(fp)
                    for d in problem['train']:
                        ti.append(np.array(d['input'],dtype=np.uint8))
                        to.append(np.array(d['output'],dtype=np.uint8))
        else:
            with open(self.train_file, 'rb') as f:
                full_list = pickle.load(f)                    
            ti = full_list[0]
            to = full_list[1]

        if not os.path.exists(self.eval_file):
            for p in self._pathlist:

                with open(p) as fp:
                    problem = json.load(fp)
                    for d in problem['test']:
                        ei.append(np.array(d['input'],dtype=np.uint8))
                        eo.append(np.array(d['output'],dtype=np.uint8))
        else:
            with open(self.eval_file, 'rb') as f:
                full_list = pickle.load(f)                    
            ei = full_list[0]
            eo = full_list[1]

        #desc = {'id': os.path.basename(fp.name).split('.')[0]}
        desc = {'id': f'{self.train_task}_{self.eval_task}'}
        dat.append((ti,to,ei,eo,desc))

    
        print(len(ti), len(to), len(ei), len(eo))
        return dat



def rotate_left(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[j][2-i])
        rotate_state.append(temp)
    return rotate_state

# rotate_right function is a clockwise rotation about the given state.
def rotate_right(state):
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[2-j][i])
        rotate_state.append(temp)
    return rotate_state

# vertical_flip function is a flip by y-axis about the given state
def vertical_flip(state): 
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[2-i][j])
        rotate_state.append(temp)
    return rotate_state

# horizontal_flip function is a flip by x-axis about the given state.
def horizontal_flip(state): 
    temp_state = copy.deepcopy(state['grid'] if 'grid' in state else state)
    rotate_state = []
    for  i in range(3):
        temp = []
        for j in range(3):
            temp.append(temp_state[i][2-j])
        rotate_state.append(temp)
    return rotate_state

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config