"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import gc
import math
import torch.nn.functional as F
import numpy as np
import torch
import time
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

GAMMA = 0.99
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
        x=x.view(x.shape[0],-1)

        #torch.Size([8, 8, 31]) torch.Size([8, 31]) self.log_prob(x)
        #log_p = self.log_prob(x).mean(axis=1)
        return self.log_prob(x).mean(axis=1).sum(axis=1) #self.log_prob(x).sum(axis=2)


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

        # self.apply(weight_init)

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
        print('dataloader',len(dataloader))
        for en, trajs in enumerate(dataloader):
            loss = self.train_step_stochastic(loss_fn, trajs)
            #self.target.load_state_dict(self.model.state_dict())
            for name, target_param in self.target.named_parameters():
                for param in self.model.state_dict():
                    if param == name:
                        target_param.data.copy_(self.model.state_dict()[param].data) if \
                        self.model.state_dict()[
                            param].shape == target_param.shape else print(
                            self.model.state_dict()[param].shape,
                            target_param.shape)
            # losses.append(loss)
            # nlls.append(nll)
            # entropies.append(entropy)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(self.losses)
        logs["training/train_loss_std"] = np.std(self.losses)
        #logs["training/nll"] = self.nlls[-1]
        #logs["training/entropy"] = self.entropy[-1]
        #logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

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

        # states = states.to(self.device)
        # actions = actions.to(self.device)
        # rewards = rewards.to(self.device)
        # dones = dones.to(self.device)
        # rtg = rtg.to(self.device)
        # timesteps = timesteps.to(self.device)
        # ordering = ordering.to(self.device)
        # padding_mask = padding_mask.to(self.device)

        # action_target = torch.clone(actions)

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
        inps = inputs_n

        for b in range(1):
            inputs = {}
            inputs_n = {}
            # inputs['observations']=inputs['observations'][0].to(self.device)
            # inputs['actions']= inputs['actions'][0].to(self.device)
            # inputs['rewards']=inputs['rewards'][0].to(self.device)
            # inputs["returns-to-go"] = inputs['returns-to-go'][0].to(self.device)
            # inputs["action_target"]=torch.clone(inputs['actions'])
            ###

            inputs['observations'] = inp['observations'][b:b + 8, :, :].to(self.device)
            inputs['actions'] = inp['actions'][b:b + 8, :].to(self.device)
            inputs['rewards'] = inp['rewards'][b:b + 8, :].to(self.device)
            inputs["returns-to-go"] = inp['returns-to-go'][b:b + 8, :].to(self.device)
            inputs["action_target"] = torch.clone(inp['actions'][b:b + 8, :])
            inputs['picked']=inp['picked'][b:b + 8, :].to(self.device)
            inputs_n['observations'] = inps['observations'][b:b + 8, :, :]#.to(self.device)
            inputs_n['actions'] = inps['actions'][b:b + 8, :]#.to(self.device)
            inputs_n['rewards'] = inps['rewards'][b:b + 8, :]#.to(self.device)
            inputs_n["returns-to-go"] = inps['returns-to-go'][b:b + 8, :]#.to(self.device)
            inputs_n["action_target"] = torch.clone(inps['actions'][b:b + 8, :])
            inputs_n['picked'] = inps['picked'][b:b + 8, :]#.to(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            #map_results = self.model.forward(inputs)
            #action_preds = map_results["action_logits"]
            #print(action_preds.shape, inputs['picked'].shape,"trainerere")
            #action_preds = torch.softmax(action_preds, dim=-1) * inputs['picked']
            actions_v = inp["actions"][b:b +8, :].to(self.device)  # torch.tensor(actions).to(device)
            done_mask = torch.BoolTensor(dones[b:b + 8]).to(device=self.device)
            state_action_values = self.model(inputs)["action_logits"]
            state_action_values=torch.softmax(state_action_values, dim=-1) * inputs['picked']
            #([8, 8, 31]) torch.Size([8, 1]) 253
            #torch.Size([384, 31, 1025]) torch.Size([384, 1]) torch.Size([384, 1])torch.Size([8, 8, 31])torch.Size([8, 1])253
            #torch.Size([8, 1])torch.Size([8, 8])263

            state_action_values= state_action_values.gather(1, actions_v).squeeze(-1).to(
                device=self.device)  # net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            #self.target = self.target.to(device=self.device)
            next_state_values = self.target(inputs_n)["action_logits"]
            next_state_values=torch.softmax(next_state_values, dim=-1) * inputs_n['picked']
            print(next_state_values.max(1)[0].shape,'261',done_mask.shape,next_state_values.shape)
            #torch.Size([8]) 261 torch.Size([8]) torch.Size([8, 31])
            next_state_values=next_state_values.max(1)[0].to(device=self.device)

            next_state_values[done_mask] = 0.0  # no discounted reward for done states
            next_state_values = next_state_values.detach()  # return the tensor without connection to its calculation history
            expected_state_action_values = next_state_values.float() * GAMMA + torch.squeeze(inputs["rewards"].float())
            print(state_action_values.shape,inputs["rewards"].shape,expected_state_action_values.shape,'263')
            #torch.Size([8]) torch.Size([8, 1]) torch.Size([8, 8]) 263
            loss = loss_fn(state_action_values, expected_state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.losses.append(loss.detach().cpu().item())

        return (
            loss.detach().cpu().item(),
        )
