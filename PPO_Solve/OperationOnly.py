from arcle.envs import O2ARCv2Env
from arcle.loaders import ARCLoader, Loader
import numpy as np

from arcle.loaders import ARCLoader
from gymnasium.core import ObsType, ActType
import numpy as np
from typing import Dict,Optional, Tuple, SupportsFloat, SupportsInt, SupportsIndex, Any
from functools import wraps


class OpOnlyenv(O2ARCv2Env):
    
    def __init__(self, data_loader: Loader = ARCLoader(), max_grid_size: Tuple[SupportsInt, SupportsInt] = (30,30), colors: SupportsInt = 10, max_trial: SupportsInt = -1, render_mode: str = None, render_size: Tuple[SupportsInt, SupportsInt] = None) -> None:
        super().__init__(data_loader, max_grid_size, colors, max_trial, render_mode, render_size)

        self.reset_options = {
            'adaptation': True, # Default is true (adaptation first!). To change this mode, call 'post_adaptation()'
            'prob_index': None
        }
        self.num_func = 5
    def create_operations(self) :
        from arcle.actions.critical import crop_grid
        from arcle.actions.object import reset_sel
        ops = super().create_operations()
        new_ops = []
        for i, op in enumerate(ops):
            if i in [24,25,26,27,34]: #[4,6,8,9,24,25,26,27, 29, 30, 21, 34]:
                new_ops.append(ops[i])
        return new_ops
    
    def reset(self, seed = None, options: Optional[Dict] = None, subprob = None):
            super().reset(seed=seed,options=options)
    
            # Reset Internal States
            self.truncated = False
            self.submit_count = 0
            self.last_action: ActType  = None
            self.last_action_op : SupportsIndex  = None
            self.last_reward: SupportsFloat = 0
            self.action_steps: SupportsInt = 0
            self.eval_subprob = subprob
            # env option
            self.prob_index = None
            self.subprob_index = None
            self.adaptation = True
            self.reset_on_submit = False
            self.options = options

            if options is not None:
                self.prob_index = options.get('prob_index')
                self.subprob_index = options.get('subprob_index')
                _ad = options.get('adaptation')
                self.adaptation = True if _ad is None else bool(_ad)
                _ros = options.get('reset_on_submit')
                self.reset_on_submit = False if _ros is None else _ros
            
            ex_in, ex_out, tt_in, tt_out, desc = self.loader.pick(data_index=self.prob_index)

            
            if self.adaptation:
                self.subprob_index = np.random.randint(0,len(ex_in)) if self.subprob_index is None else self.subprob_index
                self.input_ = ex_in[self.subprob_index]
                self.answer = ex_out[self.subprob_index]

            else: #eval_problem 1 to 100
                self.subprob_index = self.eval_subprob
                self.input_ = tt_in[self.subprob_index]
                self.answer = tt_out[self.subprob_index]

            self.init_state(self.input_.copy(),options)

            self.description = desc

            if self.render_mode:
                self.render()

            obs = self.current_state
            self.info = self.init_info()

            return obs, self.info
    
    def reward(self, state) -> SupportsFloat:
        if not self.last_action_op == len(self.operations)-1:
            return 0
        if tuple(state['grid_dim']) == self.answer.shape:
            h,w = self.answer.shape
            if np.all(state['grid'][0:h, 0:w] == self.answer):
                return 1
        return 0
    
    def step(self, action: ActType):

        operation = int(action['operation'])

        self.transition(self.current_state, action)
        self.last_action_op = operation
        self.last_action = action

        # do action
        state = self.current_state
        reward = self.reward(state)
        self.last_reward = reward
        self.action_steps+=1
        self.info['steps'] = self.action_steps
        self.info['submit_count'] = self.submit_count
        self.render()

        return self.current_state, reward, bool(state["terminated"][0]), self.truncated, self.info

    def transition(self, state: ObsType, action: ActType) -> None:
        op = int(action['operation'])
        self.last_action_op = op
        self.last_action = action
        self.operations[op](state,action)