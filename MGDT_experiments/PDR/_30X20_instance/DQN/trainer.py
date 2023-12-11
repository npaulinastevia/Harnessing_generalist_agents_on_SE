"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import torch.distributions as D
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
            target,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.target = target
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()
        self.gamma = 0.99

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        self.losses, self.nlls, self.entropy = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        self.target.train()
        for en, trajs in enumerate(dataloader):

            loss= self.train_step_stochastic(loss_fn, trajs)
            for name, target_param in self.target.named_parameters():
                for param in self.model.state_dict():
                    if param == name:
                        target_param.data.copy_(self.model.state_dict()[param].data) if \
                            self.model.state_dict()[
                                param].shape == target_param.shape else print(
                            self.model.state_dict()[param].shape,
                            target_param.shape)
            #self.model.load_state_dict(self.target.state_dict())
            #losses.append(loss)
            #nlls.append(nll)
            #entropies.append(entropy)

            
    
           
           
        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(self.losses)
        logs["training/train_loss_std"] = np.std(self.losses)

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
            inputs_n
        ) = trajs


        inp = inputs
        inps = inputs_n
        dn = dones
        #print(inp['adj'].shape,inputs['fea'].shape,inputs['candidate'].shape,inputs['mask'].shape,'inp[adj].shape',)
        for b in range(1):
            inputs = {}
            input_n = {}
            inputs['adj'] = inp['adj'][b:b + 32, :, :].to(self.device)
            mb_g_pool = g_pool_cal(configs.graph_pool_type, inputs['adj'].to(device=self.device).shape, configs.n_j * configs.n_m,
                                   self.device)
            temp=aggr_obs(torch.stack(tuple(x for x in inputs['adj'])).to(device=self.device), configs.n_j*configs.n_m).to(device=self.device)

            inputs['observations'] = inp['observations'][b:b + 32].to(self.device)
            inputs['actions'] = inp['actions'][b:b+32].to(self.device)
            inputs['rewards'] = inp['rewards'][b:b+32].to(self.device)
            inputs['idx'] = inp['idx'][b:b + 32].to(self.device)
            inputs["returns-to-go"] = inp['returns-to-go'][b:b+32,:].to(self.device)
            inputs["action_target"] = torch.clone(inp['actions'][b:b+32])
            #inputs['stateval']=inp['stateval'][b:b+32,:,:].to(self.device)
            #inputs['actlogprob']=inp['actlogprob'][b:b+32].to(self.device)
            #inputs['adj'] = inp['adj'][b:b+32,:,:].view(-1,inputs['adj'].shape[-1]).to(self.device)
            inputs['fea'] = inp['fea'][b:b+32,:,:].to(self.device)
            inputs["mask"] = inp['mask'][b:b+32,:].to(self.device)
            inputs["candidate"] = inp['candidate'][b:b+32,:].to(self.device)
            inputs['adj'] = temp
            inputs['fea'] = inputs['fea'].view(-1, inputs['fea'].shape[-1]).to(self.device)


            inputs_n['adj'] = inps['adj'][b:b + 32, :, :].to(self.device)

            temp=aggr_obs(torch.stack(tuple(x for x in inputs_n['adj'])).to(device=self.device), configs.n_j*configs.n_m).to(device=self.device)

            #inputs_n['observations'] = inps['observations'][b:b + 32].to(self.device)
            inputs_n['actions'] = inps['actions'][b:b+32].to(self.device)
            inputs_n['rewards'] = inps['rewards'][b:b+32].to(self.device)
            inputs_n["returns-to-go"] = inps['returns-to-go'][b:b+32,:].to(self.device)
            inputs_n["action_target"] = torch.clone(inps['actions'][b:b+32])
            inputs_n['observations'] = inp['observations'][b:b + 32].to(self.device)
            #inputs_n['stateval']=inps['stateval'][b:b+32].to(self.device)
            #inputs_n['actlogprob']=inps['actlogprob'][b:b+32].to(self.device)
            #inputs['adj'] = inp['adj'][b:b+32,:,:].view(-1,inputs['adj'].shape[-1]).to(self.device)
            inputs_n['fea'] = inps['fea'][b:b+32,:,:].to(self.device)
            inputs_n["mask"] = inps['mask'][b:b+32,:].to(self.device)
            inputs_n["candidate"] = inps['candidate'][b:b+32,:].to(self.device)
            inputs_n['adj'] = temp
            inputs_n['fea'] = inputs_n['fea'].view(-1, inputs_n['fea'].shape[-1]).to(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            #dones = torch.BoolTensor(dn[b:b + 32]).to(device=self.device)
            rewards = []

            self.K_epochs=1
            for _ in range(self.K_epochs):
                actions_v = inp["idx"][b:b + 32].to(self.device)  # torch.tensor(actions).to(device)  actions_v = inp["actions"][b:b + 32].to(self.device)
                done_mask = torch.BoolTensor(dn[b:b + 32]).to(device=self.device)
                xxx=self.model.forward(inputs['fea'],
                                                      mb_g_pool,
                                                      None,
                                                      inputs['adj'],
                                                      inputs['candidate'],
                                                      inputs['mask'], inputs)["action_logits"]

                state_action_values = torch.squeeze(xxx).gather(0, actions_v.unsqueeze(-1)).squeeze(
                    -1).to(
                    device=self.device)  # net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                self.target = self.target.to(device=self.device)
                next_state_values = self.target.forward(inputs_n['fea'],
                                                      mb_g_pool,
                                                      None,
                                                      inputs_n['adj'],
                                                      inputs_n['candidate'],
                                                      inputs_n['mask'], inputs_n)["action_logits"].max(1)[0].to(device=self.device)
                # print(done_mask, 'done_mask',next_state_values.shape,state_action_values.shape)
                # tensor([[False, False, False, False, False, False, False, False, False, False,
                # False, False, False, False, False, False, False, False, False, False]],
                # device='cuda:0') done_mask torch.Size([20, 4]) torch.Size([20, 4])

                next_state_values[done_mask[0]] = 0.0  # no discounted reward for done states
                next_state_values = next_state_values.detach()  # return the tensor without connection to its calculation history

                expected_state_action_values = next_state_values.float() * self.gamma + inputs["rewards"].float()
                loss = loss_fn(state_action_values, expected_state_action_values)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.model = self.model.to(device=self.device)
                self.optimizer.step()
                self.losses.append(loss.detach().cpu().item())
                ###icici

             



        return (
            loss.mean().detach().cpu().item(),

        )
