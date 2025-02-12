"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.total_length = (config.grid_x*config.grid_y+4)*config.context_length
        
        self.register_buffer("mask", torch.tril(torch.ones(self.total_length, self.total_length))
                                     .view(1, 1, self.total_length, self.total_length))
        
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.num_pixel = config.grid_x * config.grid_y
        
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.head_state = nn.Linear(config.n_embd, config.color_num)
        self.head_action = nn.Linear(config.n_embd, config.vocab_size)
        self.head_rtg = nn.Linear(config.n_embd, 1)
        self.head_intention = nn.Linear(config.n_embd, 2 * config.intention_size)

        
        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_encoder = nn.Sequential(
            nn.Embedding(config.color_num, config.n_embd),
        )
        
        self.action_embeddings = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd), 
        )
        
        self.ret_emb = nn.Sequential(
            nn.Linear(1, config.n_embd), 
        )
        
        self.pnp_emb = nn.Sequential(
            nn.Linear(self.num_pixel, config.n_embd),
        )

        self.src_emb = nn.Sequential(
            nn.Embedding(config.intention_size, config.n_embd),
        )
        self.dst_emb = nn.Sequential(
            nn.Embedding(config.intention_size, config.n_embd),
        )

        self.intention_emb = nn.Sequential(
            nn.Linear(2 * config.n_embd, config.n_embd)
        )

        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    decay.add(fpn)    
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        decay.add('pos_emb')
        decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        
        return optimizer

    # state, action, and return
    def forward(self, states, actions, rtgs=None, timesteps=None, pnp=None, intentions=None, targets=False):

        rtgs = rtgs.unsqueeze(-1)

        if targets:
            state_embeddings = self.state_encoder(states[:,:-self.num_pixel].type(torch.long))
            action_embeddings = self.action_embeddings(actions[:,:-1].type(torch.long)) 
            rtg_embeddings = self.ret_emb(rtgs[:,:-1].type(torch.float32))
            pnp_embeddings = self.pnp_emb(pnp[:,:-1].type(torch.float32))
            src_emb = self.src_emb(intentions[:,:-1, 0].type(torch.long))
            dst_emb = self.dst_emb(intentions[:,:-1, 1].type(torch.long))
        else:
            state_embeddings = self.state_encoder(states.type(torch.long))
            action_embeddings = self.action_embeddings(actions.type(torch.long)) 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            pnp_embeddings = self.pnp_emb(pnp.type(torch.float32))
            src_emb = self.src_emb(intentions[..., 0].type(torch.long))
            dst_emb = self.dst_emb(intentions[..., 1].type(torch.long))

        intention_embeddings = self.intention_emb(torch.cat((src_emb, dst_emb), dim=-1))
    
        token_embeddings = torch.zeros(
            (actions.shape[0], (action_embeddings.shape[1]) * (self.num_pixel+4), self.config.n_embd), 
            dtype=torch.float32, device=action_embeddings.device
        )

        token_embeddings[:,0::self.num_pixel+4,:] = rtg_embeddings[:,:,:]
        for i in range(self.num_pixel):
            token_embeddings[:,1+i::self.num_pixel+4,:] = state_embeddings[:,i::self.num_pixel,:]
        token_embeddings[:,self.num_pixel+1::self.num_pixel+4,:] = action_embeddings[:,:,:]
        token_embeddings[:,self.num_pixel+2::self.num_pixel+4,:] = pnp_embeddings[:,:,:]
        token_embeddings[:,self.num_pixel+3::self.num_pixel+4,:] = intention_embeddings[:,:,:]

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        
        if targets:
            position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave((timesteps[:,:-1]).unsqueeze(-1), self.config.n_embd, dim=-1).type(torch.int64)) + torch.repeat_interleave(self.pos_emb[:, :self.pos_emb.shape[1]-1, :], batch_size, dim=0)
        else:
            position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave((timesteps[:,:]).unsqueeze(-1), self.config.n_embd, dim=-1).type(torch.int64)) + torch.repeat_interleave(self.pos_emb[:, :self.pos_emb.shape[1]-1, :], batch_size, dim=0)
            
        position_embeddings = torch.repeat_interleave(position_embeddings, self.num_pixel+4, dim=1).contiguous()

        # Transformer forward
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        # Output processing
        logits = self.head_state(x).squeeze(-1)
        logit_action = self.head_action(x[:, self.num_pixel+1::self.num_pixel+4]).squeeze(-1)
        logit_rtg = self.head_rtg(x[:, 0::self.num_pixel+4]).squeeze(-1)
        logit_src = None
        logit_dst = None
        logit_intention = self.head_intention(x[:, self.num_pixel+3::self.num_pixel+4]).squeeze(-1)
    
        batch_size = logit_intention.shape[0]
        logit_src = logit_intention[:, :, :self.config.intention_size].view(batch_size, -1, self.config.intention_size)
        logit_dst = logit_intention[:, :, self.config.intention_size:].view(batch_size, -1, self.config.intention_size)

        logit_state = torch.zeros(
            (logits.shape[0], (logit_action.shape[1]) * self.num_pixel, logits.shape[2]), 
            dtype=torch.float32, device=states.device
        )

        for i in range(self.num_pixel):
            logit_state[:, i::self.num_pixel] = logits[:, i+1::self.num_pixel+4]

        loss = None
        loss_s = None
        loss_a = None
        loss_r = None
        loss_i = None

        if targets:
            criterion = nn.CrossEntropyLoss()
            mse_loss = nn.MSELoss()

            loss_s = criterion(logit_state.reshape(-1,logit_state.shape[-1]), (states[:,self.num_pixel:]).reshape(-1))
            loss_a = criterion(logit_action.reshape(-1,logit_action.shape[-1]), (actions[:,1:]).reshape(-1))     
            loss_r = mse_loss(logit_rtg.reshape(-1), rtgs[:,1:].reshape(-1))

            batch_size, seq_len, intention_size = logit_src.shape
            logit_src = logit_src.reshape(batch_size * seq_len, intention_size)
            logit_dst = logit_dst.reshape(batch_size * seq_len, intention_size)

            target_intention = intentions[:, 1:, :].reshape(-1, 2)  # (batch, seq_len-1, 2) -> (batch*(seq_len-1), 2)
            target_src = target_intention[:, 0]  # source intention (batch*(seq_len-1))
            target_dst = target_intention[:, 1]  # destination intention (batch*(seq_len-1))

            loss = loss_s + loss_a + loss_r

            loss_src = criterion(logit_src, target_src)  # source loss
            loss_dst = criterion(logit_dst, target_dst)  # destination loss
            loss_i = loss_src + loss_dst  

            loss += loss_i

        return logit_state, logit_action, logit_rtg, logit_src, logit_dst, loss, loss_s, loss_a, loss_r, loss_i