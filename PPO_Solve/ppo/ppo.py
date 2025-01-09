
import numpy as np

from .model import get_train_fn, get_act_fn
from .runner import Runner
from .policy import GPTPolicy
from utils.util import get_device
import torch
from tqdm import trange
import wandb
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
from collections import OrderedDict
from solver import solve


def learn(cfg, env):

    tcfg = cfg.train
    device = get_device(tcfg.gpu_num - 1)
    policy = GPTPolicy(cfg, device).to(device)
    # if you want to train continuelly, you can load the model using the following code
    # model_state = torch.load("/home/jovyan/ppo/data/task241-300k/saved_parameter_90.pth")
    # new_model_state = OrderedDict([(key.replace('_orig_mod.', ''), value) for key, value in model_state.items()])
    # policy.load_state_dict(new_model_state)
    policy = torch.compile(policy)

    nenvs = tcfg.nenvs
    nbatch = nenvs * tcfg.nsteps

    nbatch_train = nbatch // tcfg.nminibatches
    nupdates = tcfg.total_timesteps // nbatch

    optimizer = policy.configure_optimizers()
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1000,
                                          cycle_mult=1.0,
                                          max_lr=5e-4,
                                          min_lr=5e-5,
                                          warmup_steps=500,
                                          gamma=1.0)    
    train_fn = get_train_fn(policy, optimizer, tcfg)
    act_fn = get_act_fn(policy)

    # Instantiate the runner object
    runner = Runner(env, cfg, device, act_fn, True)
    all_ep_rets = []

    for update in trange(1, nupdates + 1):
        assert nbatch % tcfg.nminibatches == 0
        steps = 0
        batch_ob_acs = []
        batch_returns = []
        batch_values = []
        batch_neglogpacs = [] 
        batch_rtm1 = []
        batch_rtm1_pred= []
        batch_norm_rew = []
        batch_rpred=[]
        batch_gpred=[]
        batch_gtp1 = []
        batch_ep_rets = []
        batch_success_ts = []
        batch_sum_reward = []
        success = 0
        count = 0
        while steps < nenvs * tcfg.nsteps:
            (ob_acs, returns, values, neglogpacs, rtm1, rtm1_pred, norm_rew, rpred, gpred, gtp1,
            ep_rets, success_ts, unwrapped, clip, input, answer, sum_reward, step) = runner.run()
            count += 1
            batch_ob_acs.append(ob_acs)
            batch_returns.append(returns)
            batch_values.append(values)
            batch_neglogpacs.append(neglogpacs)
            batch_rtm1.append(rtm1)
            batch_rtm1_pred.append(rtm1_pred)
            batch_norm_rew.append(norm_rew)
            batch_rpred.append(rpred)
            batch_gpred.append(gpred)
            batch_gtp1.append(gtp1)
            batch_ep_rets.append(ep_rets)
            batch_success_ts.append(success_ts)
            batch_sum_reward.append(sum_reward)
            steps += step
            if sum_reward > 1000:
                success += 1

        print("step", steps)
        print("episode_num", count)
        batch_ob_acs_dic = {}
        for key in ob_acs.keys():
            batch_ob_acs_dic[key] = []
        for i, item in enumerate(batch_ob_acs):
            for key in item.keys():
                batch_ob_acs_dic[key].append(batch_ob_acs[i][key])
        for key in ob_acs.keys():
            batch_ob_acs_dic[key] = torch.cat(batch_ob_acs_dic[key], dim=0)

            
        batch_neglogpacs = torch.cat(batch_neglogpacs, dim=0)
        batch_rtm1 = torch.cat(batch_rtm1, dim=0)
        batch_rtm1_pred = torch.cat(batch_rtm1_pred, dim=0)
        batch_norm_rew = torch.cat(batch_norm_rew, dim=0)
        batch_rpred = torch.cat(batch_rpred, dim=0)
        batch_gpred = torch.cat(batch_gpred, dim=0)
        batch_gtp1 = torch.cat(batch_gtp1, dim=0)
        batch_returns = torch.cat(batch_returns, dim = 0)
        batch_values = torch.cat(batch_values, dim = 0)
        batch_sum_reward.append(sum_reward)

        advs = batch_returns - batch_values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        all_ep_rets += batch_ep_rets
        print("Done.")
        # Here what we're going to do is for each minibatch calculate the loss and append it.
        train_rets = []
        
        # Index of each element of batch_size
        # Create the indices array
        print("Training:")
        inds = np.arange(nbatch) # 3200

        train_actor = update >= tcfg.update_actor_after

        for _ in trange(tcfg.noptepochs):
            # Randomize the indexes

            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                ob_ac_slice = {key: batch_ob_acs_dic[key][mbinds] for key in batch_ob_acs_dic}
                slices = (arr[mbinds] for arr in (
                    batch_returns, batch_values, batch_neglogpacs, advs, batch_rtm1, batch_norm_rew, batch_gtp1))
                train_rets.append(torch.stack(train_fn(ob_ac_slice, *slices, train_actor)))
        print("Done.")
        scheduler.step()

        train_accuracy = success / count
        timestep = nenvs * tcfg.nsteps * update


        # Feedforward --> get losses --> update
        lossvals = npy(torch.mean(torch.stack(train_rets), axis=0))

        if update % tcfg.log_interval == 0:
            watched = {"old_state_val": batch_values, 
                       "old_neg_logprobs": batch_neglogpacs, 
                       "returns": batch_returns, 
                       "rewards": batch_rtm1, 
                       "rtm1_pred": batch_rtm1_pred, 
                       "rpred": batch_rpred}
            logged = {}
            for key, val in watched.items():
                logged[f"{key}/mean"] = safemean(npy(val))
                logged[f"{key}/min"] = np.min(npy(val))
                logged[f"{key}/max"] = np.max(npy(val))
                logged[f"{key}/std"] = np.std(npy(val))
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = max(1 - np.var(npy(returns) - npy(values))/np.var(npy(returns)), -1)
            ev_aux_rtm1 = max(1 - np.var(npy(rtm1) - npy(rtm1_pred))/np.var(npy(rtm1)), -1)
            ev_aux_rew = max(1 - np.var(npy(norm_rew) - npy(rpred))/np.var(npy(norm_rew)), -1)
            ev_aux_rew_on_rtm1 = max(1 - np.var(npy(norm_rew) - npy(rpred))/np.var(npy(norm_rew) - npy(rtm1_pred)), -1)
            acc_aux_gtp1 = (npy(gpred).argmax(2) == npy(gtp1)).mean()
            # wandb.log(
            #     {
            #         "rollout/operation": npy(batch_ob_acs_dic["operation"]),
            #         "misc/serial_timesteps": update * tcfg.nsteps,
            #         "misc/nupdates": update,
            #         "misc/total_timesteps": update * nbatch, 
            #         "explained_variance": float(ev),
            #         "explained_variance_aux_rtm1": float(ev_aux_rtm1),
            #         "explained_variance_aux_rew": float(ev_aux_rew),
            #         "explained_variance_aux_rew_on_rtm1": float(ev_aux_rew_on_rtm1),
            #         "acc_aux_gtp1": float(acc_aux_gtp1),
            #         'eprewmean': safemean(all_ep_rets),
            #         'success_ts': safemean(batch_success_ts),
            #         'success_rate': safemean(np.array(batch_success_ts) > 0),
            #         'lr': scheduler.get_lr()[0],
            #         'loss/loss': lossvals[0],
            #         'loss/pg_loss': lossvals[1],
            #         'loss/vf_loss': lossvals[2],
            #         'loss/entropy_loss': lossvals[3],
            #         'loss/aux_loss_rtm1': lossvals[4],
            #         'loss/aux_loss_rew': lossvals[5],
            #         'loss/aux_loss_gtp1': lossvals[6],
            #         'loss/approxkl': lossvals[7],
            #         'loss/clipfrac': lossvals[8],
            #         'train_accuracy': train_accuracy,
            #     } | logged,
            #     step=timestep
            # )
            all_ep_rets = []
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'data/test1')

            if update % 1 == 0:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)     
                    torch.save(policy.state_dict(), "{}/saved_parameter_{}.pth".format(model_path, update))
                else:
                    torch.save(policy.state_dict(), "{}/saved_parameter_{}.pth".format(model_path, update))

                _, _, _, _, _, accuarcy  = solve(cfg, env, update, model_path)
                # wandb.log(
                #     {  
                #         "eval_accuarcy": accuarcy,
                #     },
                #     step=timestep
                # )

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def npy(tensor):
    return tensor.detach().cpu().numpy()
