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

GAMMA = 0.99
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
        self.gamma=0.99
        self.K_epochs=15

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

          
            loss = self.train_step_stochastic(loss_fn, trajs)
            #losses.append(torch.mean(loss))
            self.model.load_state_dict(self.target.state_dict())
            #self.target.load_state_dict(self.model.state_dict())
         
        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(self.losses)
        logs["training/train_loss_std"] = np.std(self.losses)

        return logs

    # def train_dqn(self):

    def train_step_stochasticdqn(self, loss_fn, trajs):
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
            inputs_n,
        ) = trajs

        # states = states.to(self.device)
        # actions = actions.to(self.device)
        # rewards = rewards.to(self.device)
        # dones = dones.to(self.device)
        # rtg = rtg.to(self.device)
        # timesteps = timesteps.to(self.device)
        # ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

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

        inputs['observations'] = inputs['observations'][0].to(self.device)
        inputs['actions'] = inputs['actions'][0].to(self.device)
        inputs['rewards'] = inputs['rewards'][0].to(self.device)
        inputs["returns-to-go"] = inputs['returns-to-go'][0].to(self.device)
        inputs["action_target"] = torch.clone(inputs['actions'])
        inputs_n['observations'] = inputs_n['observations'][0].to(self.device)
        inputs_n['actions'] = inputs_n['actions'][0].to(self.device)
        inputs_n['rewards'] = inputs_n['rewards'][0].to(self.device)
        inputs_n["returns-to-go"] = inputs_n['returns-to-go'][0].to(self.device)
        inputs_n["action_target"] = torch.clone(inputs_n['actions'])
        gc.collect()
        torch.cuda.empty_cache()
        actions_v = inputs["actions"]  # torch.tensor(actions).to(device)
        done_mask = torch.BoolTensor(dones).to(device=self.device)

        state_action_values = self.model(inputs)["action_logits"].gather(2, actions_v.unsqueeze(-1)).squeeze(-1).to(
            device=self.device)  # net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        self.target=self.target.to(device=self.device)
        next_state_values = self.target(inputs_n)["action_logits"].max(2)[0].to(device=self.device)
        next_state_values[done_mask[0]] = 0.0  # no discounted reward for done states
        next_state_values = next_state_values.detach()  # return the tensor without connection to its calculation history

        expected_state_action_values = next_state_values.float() * GAMMA + inputs["rewards"].float()
        loss = loss_fn(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        return (
            loss.detach().cpu().item()

        )

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewar,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
            inputs,
  
        ) = trajs

        # states = states.to(self.device)
        # actions = actions.to(self.device)
        # rewards = rewards.to(self.device)
        # dones = dones.to(self.device)
        # rtg = rtg.to(self.device)
        # timesteps = timesteps.to(self.device)
        # ordering = ordering.to(self.device)
        #padding_mask = padding_mask.to(self.device)

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
        dn=dones

        for b in range(1):
            inputs = {}
            inputs['observations'] = inp['observations'][b:b+32,:,:,:,:].to(self.device)
            inputs['actions'] = inp['actions'][b:b+32,:].to(self.device)
            inputs['rewards'] = inp['rewards'][b:b+32,:].to(self.device)
            inputs["returns-to-go"] = inp['returns-to-go'][b:b+32,:].to(self.device)
            inputs["action_target"] = torch.clone(inp['actions'][b:b+32,:])
            inputs['stateval']=inp['stateval'][b:b+32,:].to(self.device)
            inputs['actlogprob']=inp['actlogprob'][b:b+32,:].to(self.device)
            dones = torch.BoolTensor(dn[b:b + 32]).to(device=self.device)
            ###
            # inputs['observations'] = inputs['observations'][0].to(self.device)
            # inputs['actions'] = inputs['actions'][0].to(self.device)
            # inputs['rewards'] = inputs['rewards'][0].to(self.device)
            # inputs["returns-to-go"] = inputs['returns-to-go'][0].to(self.device)
            # inputs["action_target"] = torch.clone(inputs['actions'])
            # inputs['stateval']=inputs['stateval'].to(self.device)
            # inputs['actlogprob']=inputs['actlogprob'].to(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            rewards = []
            discounted_reward = 0
            self.eps_clip=0.2
            #print('trainer',dones.shape,inputs['observations'].shape,inputs['stateval'].shape,inputs['actlogprob'].shape)
            for reward, is_terminal in zip(reversed(inputs['rewards'].cpu().numpy()), reversed(dones.cpu().numpy())):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
    
            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    
            # convert list to tensor
    
    
            #old_state_values = inputs['stateval'].detach()
            # calculate advantages
    

            self.target = self.target.to(device=self.device)
            for _ in range(self.K_epochs):

                # Evaluating old actions and values
                map_results = self.target.forward(inputs)
                print(map_results['act_val'].shape,inputs['stateval'].shape,rewards.shape)
                advantages = rewards.detach() - map_results['act_val'].detach().view(rewards.shape)
                action_preds = map_results["action_logits"]
                # print('inputs us', inputs['observations'].shape, inputs["returns-to-go"].shape,
                #     inputs['actions'].shape, inputs['rewards'].shape, inputs["returns-to-go"].shape,action_preds.shape, padding_mask.shape,type(map_results['custom_causal_mask']),map_results['custom_causal_mask'].shape)
                #predict = DiagGaussianActor(1280, 18)
                (_, logprobs,dist_entropy) = sample_from_logits(
                    action_preds,
                    generator=torch.Generator(),
                    deterministic=False,
                    temperature=1.0,
                    top_percentile=50,
                )
    
                #predict = DiagGaussianActor(1280, 18)
                #action_preds = predict(action_preds)
                #dist_entropy=action_preds.entropy().mean()
                state_values=map_results["act_val"]
                # match state_values tensor dimensions with rewards tensor
                #state_values = torch.squeeze(state_values)
                # Finding the ratio (pi_theta / pi_theta__old)
    
                ratios = torch.exp(logprobs- inputs['actlogprob'].view(logprobs.shape))
    
                # Finding Surrogate Loss
                #advantages=advantages.view(-1)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
    
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) +  nn.MSELoss()(state_values.view(rewards.shape), rewards) - 0.01 * dist_entropy
    
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                self.losses.append(loss.mean().detach().cpu().item())

                #print('olffs', torch.mean(old_states.grad,dim=0).shape)

    
    
            # Copy new weights into old policy
    
            #padding_mask = torch.squeeze(torch.from_numpy(map_results['custom_causal_mask']), -1)
    
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)


        return (
            loss.detach().cpu()
        )

    def train_step_stochastic_sac(self, loss_fn, trajs):
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

        # states = states.to(self.device)
        # actions = actions.to(self.device)
        # rewards = rewards.to(self.device)
        # dones = dones.to(self.device)
        # rtg = rtg.to(self.device)
        # timesteps = timesteps.to(self.device)
        # ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

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

        inputs['observations'] = inputs['observations'][0].to(self.device)
        inputs['actions'] = inputs['actions'][0].to(self.device)
        inputs['rewards'] = inputs['rewards'][0].to(self.device)
        inputs["returns-to-go"] = inputs['returns-to-go'][0].to(self.device)
        inputs["action_target"] = torch.clone(inputs['actions'])
        gc.collect()
        torch.cuda.empty_cache()
        tempera = self.model.temperature()
        entro = self.model.target_entropy

        map_results = self.model.forward(inputs)
        action_preds = map_results["action_logits"]
        # print('inputs us', inputs['observations'].shape, inputs["returns-to-go"].shape,
        #     inputs['actions'].shape, inputs['rewards'].shape, inputs["returns-to-go"].shape,action_preds.shape, padding_mask.shape,type(map_results['custom_causal_mask']),map_results['custom_causal_mask'].shape)
        predict = DiagGaussianActor(1280, 4)
        action_preds = predict(action_preds)
        b = map_results['act_target']
        action_target = b[:, :, :4]
        padding_mask = torch.squeeze(torch.from_numpy(map_results['custom_causal_mask']), -1)
        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            tempera.detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        print('entropu', self.model.target_entropy, self.model.parameters(), self.model.temperature())
        temperature_loss = (
                tempera * (entropy - entro).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
