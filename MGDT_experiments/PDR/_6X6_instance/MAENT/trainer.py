"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import gc
from Params import configs
import math
import torch.nn.functional as F
import numpy as np
import torch
import time
from mb_agg import *
from agent_utils import *
from typing import Mapping, Optional, Tuple
from torch import distributions as pyd
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch import Tensor

from multigame_dt_utils import (
    accuracy,
    autoregressive_generate,
    cross_entropy,
    decode_return,
    encode_return,
    encode_reward,
    sample_from_logits,
    variance_scaling_,
)
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)

class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim=1280, act_dim=18, log_std_bounds=[-5.0, 2.0]):
        super().__init__()
        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        #self.apply(weight_init)

    def forward(self, mu):
        mu, log_std = mu, mu
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)
class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        self.losses, self.nlls, self.entropy = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for en, trajs in enumerate(dataloader):

            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            #losses.append(loss)
            #nlls.append(nll)
            #entropies.append(entropy)
            
    
           
           
        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(self.losses)
        logs["training/train_loss_std"] = np.std(self.losses)
        logs["training/nll"] = self.nlls[-1]
        logs["training/entropy"] = self.entropy[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
            inputs,
        ) = trajs

        #states = states.to(self.device)
        #actions = actions.to(self.device)
        #rewards = rewards.to(self.device)
        #dones = dones.to(self.device)
        #rtg = rtg.to(self.device)
        #timesteps = timesteps.to(self.device)
        #ordering = ordering.to(self.device)
        #padding_mask = padding_mask.to(self.device)

        #action_target = torch.clone(actions)

       # _, action_preds, _ = self.model.forward(
       #     states,
       #     actions,
       #     rewards,
        #    rtg[:, :-1],
        #    timesteps,
         #   ordering,
       #     padding_mask=padding_mask,
       # )
        inp = inputs
        print(inp['adj'].shape,inputs['fea'].shape,inputs['candidate'].shape,inputs['mask'].shape,'inp[adj].shape',)
        for b in range(1):
            inputs = {}
            # inputs['observations']=inputs['observations'][0].to(self.device)
            # inputs['actions']= inputs['actions'][0].to(self.device)
            # inputs['rewards']=inputs['rewards'][0].to(self.device)
            # inputs["returns-to-go"] = inputs['returns-to-go'][0].to(self.device)
            # inputs["action_target"]=torch.clone(inputs['actions'])[si : si + self.max_len,:,:]
            ###
            # {'observations': (37,),
            # 'actions2': (37,),
            # 'rewards': (37,),
            # 'terminals': (37,),
            # 'rewards2': (37,),
            # 'actions': (37,),
            # 'returns-to-go': (37, 1),
            # 'adj': (37, 36, 36),
            # 'fea': (37, 36, 2),
            # 'mask': (37, 6),
            # 'candidate': (37, 6)}
            inputs['adj'] = inp['adj'][b:b + 32, :, :].to(self.device)
            mb_g_pool = g_pool_cal(configs.graph_pool_type, inputs['adj'].to(device=self.device).shape, configs.n_j * configs.n_m,
                                   self.device)
            temp=aggr_obs(torch.stack(tuple(x for x in inputs['adj'])).to(device=self.device), configs.n_j*configs.n_m).to(device=self.device)

            inputs['observations'] = inp['observations'][b:b + 32].to(self.device)
            inputs['actions'] = inp['actions'][b:b+32].to(self.device)
            inputs['rewards'] = inp['rewards'][b:b+32].to(self.device)
            inputs["returns-to-go"] = inp['returns-to-go'][b:b+32,:].to(self.device)
            inputs["action_target"] = torch.clone(inp['actions'][b:b+32])
            #inputs['adj'] = inp['adj'][b:b+32,:,:].view(-1,inputs['adj'].shape[-1]).to(self.device)
            inputs['fea'] = inp['fea'][b:b+32,:,:].to(self.device)
            inputs["mask"] = inp['mask'][b:b+32,:].to(self.device)
            inputs["candidate"] = inp['candidate'][b:b+32,:].to(self.device)
            inputs['adj'] = temp
            inputs['fea'] = inputs['fea'].view(-1, inputs['fea'].shape[-1]).to(self.device)
            print(inputs['adj'].shape, inputs["candidate"].shape, inputs['fea'].shape, inputs["mask"].shape,
                  temp.shape,'shapppe' )
            gc.collect()
            torch.cuda.empty_cache()
            map_results= self.model.forward(inputs['fea'],
                                                  mb_g_pool,
                                                  None,
                                                  inputs['adj'],
                                                  inputs['candidate'],
                                                  inputs['mask'], inputs)
            action_preds=map_results["action_logits"]
            #print('inputs us', inputs['observations'].shape, inputs["returns-to-go"].shape,
             #     inputs['actions'].shape, inputs['rewards'].shape, inputs["returns-to-go"].shape,action_preds.shape, padding_mask.shape,type(map_results['custom_causal_mask']),map_results['custom_causal_mask'].shape)
            predict = DiagGaussianActor(1280, 1)
            action_preds = predict(action_preds)
            b=map_results['act_target']
            action_target = b[:, :, :1]
            padding_mask=torch.squeeze(torch.from_numpy(map_results['custom_causal_mask']),-1)
            loss, nll, entropy = loss_fn(
                action_preds,  # a_hat_dist
                action_target,
                padding_mask,
                self.model.temperature().detach(),  # no gradient taken here
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (
                self.model.temperature() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            self.losses.append(loss.detach().cpu().item())
            self.nlls.append(nll.detach().cpu().item())
            self.entropy.append(entropy.detach().cpu().item())


        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
