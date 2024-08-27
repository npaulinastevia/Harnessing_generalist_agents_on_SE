# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pprint
#import threading
import time
import timeit
import traceback
import typing

import pandas as pd
from gym.spaces import Box, Discrete
from pathlib import Path
#os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

#os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from Environment import LTREnvV2

from core import environment
from core import file_writer
from core import prof
from core import vtrace
import numpy as np

# yapf: disable

# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

# --- Create environments

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)
def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    actions = torch.clamp(actions, 0, 30)
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
        stepsofar,
    actor_index: int,
    free_queue,
    full_queue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):

    gym_env = create_env(flags)
    seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
    gym_env.seed(seed)
    env = environment.Environment(gym_env)
    env_output = env.initial()

    agent_state = model.initial_state(batch_size=1)

    agent_output, unused_state = model(env_output, agent_state)
    
    ep = 0
    if unused_state is None:
        return ep, buffers, initial_agent_state_buffers
    index=actor_index
    for key in env_output:
        if key == 'info':
            continue
        buffers[key][index][0, ...] = env_output[key]  # = env_output[key]#buffers[key][index][0, ...] = env_output[key]
    for key in agent_output:
        buffers[key][index][0, ...] = agent_output[key]
    for i, tensor in enumerate(agent_state):
        initial_agent_state_buffers[index][i][...] = tensor
    t = 0
    for r in range(20):
        if r >0:
            if ep<30:
                env = environment.Environment(gym_env)
                env_output = env.initial()
                agent_state = model.initial_state(batch_size=1)
            else:
                break
        while True:
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                if agent_state is None:
                    break
                env_output = env.step(agent_output["action"])
                for key in env_output:
                    if key=='info':
                        continue
                    if key=='frame':
                        buffers[key][index][t + 1, ...]= env_output[key]
                    if key=='reward' or key=='done' or key=='episode_return' or key=='episode_step' or key=='last_action':
                        buffers[key][index][t + 1, ...] = env_output[key]
                    if key=='picked':
                        buffers[key][index][t + 1, ...] = env_output[key]
                    #print(buffers[key][index][t + 1, ...].shape,key,env_output[key].shape)
                    #buffers[key][index][t + 1, ...] = env_output[key]

                for key in agent_output:
                    #print(buffers[key][index][t + 1, ...].shape, key, agent_output[key].shape)
                    if key=='policy_logits':
                        buffers[key][index][t + 1, ...]= agent_output[key]
                    if key=='baseline' or key=='action':
                        buffers[key][index][t + 1, ...] = agent_output[key]
                    #buffers[key][index][t + 1, ...] = agent_output[key]
                ep = ep + 1
                stepsofar=stepsofar+ep
                t=t+1
                if ep>=30:
                    break
                if stepsofar>flags.total_steps:
                    break
                if env_output['done']:
                    break





    return ep,buffers,initial_agent_state_buffers


def get_batch(
    flags,
    step,
    full_queue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,

):
    indices = np.random.choice(step+1, size=flags.batch_size, replace=True)
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }

    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )

    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,

):
    """Performs a learning (optimization) step."""
    #with lock:
    learner_outputs, unused_state = model(batch, initial_agent_state)
    if unused_state is None:
        return None, actor_model, optimizer, scheduler
    bootstrap_value = learner_outputs["baseline"][-1]
    batch = {key: tensor[1:] for key, tensor in batch.items()}
    learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

    rewards = batch["reward"]
    if flags.reward_clipping == "abs_one":
        clipped_rewards = torch.clamp(rewards, -1, 1)
    elif flags.reward_clipping == "none":
        clipped_rewards = rewards

    discounts = (~batch["done"]).float() * flags.discounting

    vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

    pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
    baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
    entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

    total_loss = pg_loss + baseline_loss + entropy_loss

    episode_returns = batch["episode_return"][batch["done"]]
    stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
    optimizer.step()
    scheduler.step()
    for name, target_param in actor_model.named_parameters():
            for param in model.state_dict():
                if param == name:
                    target_param.data.copy_(model.state_dict()[param].data) if \
                    model.state_dict()[
                        param].shape == target_param.shape else print(
                        model.state_dict()[param].shape,
                        target_param.shape)
        #actor_model.load_state_dict(model.state_dict())
    return stats,actor_model,optimizer,scheduler


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    T=30
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),

        picked=dict(size=(T + 1, num_actions), dtype=torch.int64),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.long),#float32 int64
        action=dict(size=(T + 1,), dtype=torch.int64),

    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]))

    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    import time

    train_start = time.time()
    remain=0
    timer=0
    if int(flags.stilltrain) > 0:
        df = pd.read_csv(flags.results_train)
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        last_row = df.dropna(subset=['time', 'episode']).iloc[-1]
        timer = last_row['time']
        remain = last_row['episode']
    flags.total_steps=int(30*((7500*int(flags.finetune[0]))/100))
    print(timer,remain,type(timer),type(remain),flags.total_steps)
   
    checkpointpath = flags.save_path+'/model.tar'
    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    #if flags.num_buffers < flags.batch_size:
    #    raise ValueError("num_buffers should be larger than batch_size")
    T = flags.unroll_length
    B = flags.batch_size
    flags.num_buffers=int((7500*int(flags.finetune[0]))/100)
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")
    flags.device = torch.device("cpu")
    env = create_env(flags)

    model = Net(env.observation_space.shape, env.action_space.n, flags.use_lstm)
    checkpoint_pretrain = torch.load('/scratch/f/foutsekh/nstevia/Harnessing_generalist_agents_on_SE/IMPALA_Pretrained/model.tar')
    for name, target_param in model.named_parameters():
        for param in checkpoint_pretrain["model_state_dict"]:
            if param==name:
                target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                           param].shape == target_param.shape  else print(
                    checkpoint_pretrain["model_state_dict"][param].shape,
                    target_param.shape)
    if int(flags.stilltrain)>0:
        print(int(flags.finetune[0]),'okakka')
        checkpoint_pretrain = torch.load(flags.save_path+'/model.tar')
        for name, target_param in model.named_parameters():
            for param in checkpoint_pretrain["model_state_dict"]:
                if param==name:
                    target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                               param].shape == target_param.shape  else print(
                        checkpoint_pretrain["model_state_dict"][param].shape,
                        target_param.shape)

    buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        initial_agent_state_buffers.append(state)
    learner_model = Net(
        env.observation_space.shape, env.action_space.n, flags.use_lstm, device=flags.device
    ).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step, stats = 0, {}
    episode=0
    step=step+remain
    while step < flags.total_steps:
            #timings.reset()
            st,buffers,initial_agent_state_buffers=act(flags,
                   step,
                episode,
                None,
                None,
                model,
                buffers,
                initial_agent_state_buffers)
            if st==0:
                continue
            batch, agent_state = get_batch(
                flags,
                episode,
                None,
                buffers,
                initial_agent_state_buffers,
                None,
            )
            step += st
            import time
            fi = open(flags.results_train, 'a+')
            fi.write(
                flags.algo + "," + flags.project_name + "," + str('none') + "," + str(
                    -train_start + time.time()+timer) + "," + str(
                    step) + "," + str('none') + '\n')
            fi.close()
            #if step>flags.batch_size:
            stats,model, optimizer,scheduler = learn(
                    flags, model, learner_model, batch, agent_state, optimizer, scheduler
                )
            if stats is None:
                continue
            episode=episode+1
            print(st,step,episode,'step episode')

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "flags": vars(flags),
                },
                checkpointpath,
            )



    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "flags": vars(flags),
        },
        checkpointpath,
    )

    def checkpoint():
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )
    checkpoint()
    #plogger.close()


def test(flags, num_episodes: int = 10):
    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint_pretrain = torch.load('/scratch/f/foutsekh/nstevia/Harnessing_generalist_agents_on_SE/IMPALA_Pretrained/model.tar')
    for name, target_param in model.named_parameters():
        for param in checkpoint_pretrain["model_state_dict"]:
            if param==name:
                target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                           param].shape == target_param.shape  else print(
                    checkpoint_pretrain["model_state_dict"][param].shape,
                    target_param.shape)
    if int(flags.finetune[0])>0:
        print(int(flags.finetune[0]),'okakka')
        checkpoint_pretrain = torch.load(flags.save_path+'/model.tar')
        for name, target_param in model.named_parameters():
            for param in checkpoint_pretrain["model_state_dict"]:
                if param==name:
                    target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                               param].shape == target_param.shape  else print(
                        checkpoint_pretrain["model_state_dict"][param].shape,
                        target_param.shape)
    start = time.time()
    st = 0
    total_e = 0
    all_map = []
    map_per_r = []
    final_top = [0, 0, 0]
    all_rr = []
    for c in range(env.gym_env.suppoerted_len):  # num_episodes // num_batch
        total_e += 1
        rew_all = 0
        irr = False
        env.returnrr = True
        all_rr.append(-100)
        t_done = False
        # total_rew = 0
        t = 0
        pik = []
        observation = env.initial()
        agent_state = model.initial_state(batch_size=1)
        while not t_done:  # for t in range(num_steps)
            agent_outputs = model(observation, agent_state)
            policy_outputs, _ = agent_outputs
            if policy_outputs["action"] is None:
                break
            observation = env.step(policy_outputs["action"])
            rew_all = rew_all + observation['reward'].item()
            if all_rr[-1] < observation['rr']:
                all_rr[-1] = observation['rr']
            #pik.append(actions.item())
            t = t + 1

            if observation["done"].item():
                t_done = True
        try:
            real_fix = 0
            precision_at_k = []
            print('next')
            print(env.topicked)
            print(env.gym_env.match_id)
            print('nextnnnn')
            for kk in range(len(env.topicked)):
                if env.topicked[kk] in env.gym_env.match_id:
                    real_fix += 1
                precision_at_k.append(real_fix / (kk + 1))
                all_map.append(precision_at_k)
            map_per_r.append(np.mean(np.array(precision_at_k)))
            temp = np.array(all_rr)
            temp = temp[temp > 0]
            top = []
            if irr:
                top = [0, 0, 0]
            else:
                for u in [1, 5, 10]:
                    cc = 0

                    for uu in range(u):

                        if len(env.topicked) >= uu + 1:
                            if env.topicked[uu] in env.gym_env.match_id:
                                cc += 1
                    top.append(cc)
            ret = all_rr[-1] if all_rr[-1] > 0 else -1
            fi = open(flags.box_test, 'a+')
            fi.write(
                flags.algo + "," + flags.project_name + "," + str(total_e) + "," + str(ret) + "," + str(
                    map_per_r[-1]) + "," + str(top[0]) +
                "," + str(top[1]) + "," + str(top[2]) + '\n')
            fi.close()
            final_top[0], final_top[1], final_top[2] = final_top[0] + top[0], final_top[1] + top[1], final_top[2] + top[
                2]
            fi = open(flags.results_test, 'a+')
            fi.write(
                flags.algo + "," + flags.project_name + "," + str(rew_all) + "," + str(
                    -start + time.time()) + "," + str(
                    total_e) + "," + str(st) + "," + str(temp.mean()) + "," + str(
                    np.mean(np.array(map_per_r))) + "," + str(final_top[0] / env.gym_env.suppoerted_len) + "," + str(
                    final_top[1] / env.gym_env.suppoerted_len) + "," + str(
                    final_top[2] / env.gym_env.suppoerted_len) + '\n')
            fi.close()
        except Exception as ex:
            print(ex)





class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False, device=None):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        #use_lstm=False
        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,#32
            kernel_size=5,#5
            stride=2,#1
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)#nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)#nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(1025)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(31)))
        linear_input_size = convw * convh * 32
        # Fully connected layer.
        self.device=device
        self.is_finetune=False
        self.use_lstm = use_lstm
        #self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = linear_input_size + num_actions +1


        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)
    def fineModel(self,is_finetune):
        if not is_finetune:
            core_output_size = self.fc.out_features + self.num_actions +1

            if self.use_lstm:
                self.core = nn.LSTM(core_output_size, core_output_size, 2,device=self.device)

            self.policy = nn.Linear(core_output_size, self.num_actions,device=self.device)
            self.baseline = nn.Linear(core_output_size, 1,device=self.device)
            self.is_finetune = True

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        x=torch.clamp(x,1e-9,31)
        T,B, *_ = x.shape
        x=x.view(-1,1,x.shape[-2],x.shape[-1])
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.Flatten()(x)#x.view(T * B, -1)
        #self.fc = nn.Linear(x.shape[1], 512,device=self.device)
       
        #self.fineModel(self.is_finetune)
        #x = F.relu(self.fc(x))
        #print(inputs["last_action"].shape,T,B,self.num_actions,'shshshkd')
        #torch.Size([2, 8]) 1 2 31 shshshkd
        inputs["last_action"]=torch.clamp(inputs["last_action"],0,self.num_actions-1)
        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions #inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 3).view(T * B, 1)
        inputs["picked"]=torch.clamp(inputs["picked"],0,self.num_actions)
        #clipped_reward = inputs["reward"].view(T * B, 1)
        temp_action = inputs["picked"].view(T * B, -1)
        #torch.Size([1, 31]) torch.Size([1, 544]) torch.Size([1, 512]) torch.Size([1, 1]) torch.Size([1, 31]) temp_action
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)#torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)

                core_state = tuple(nd * s for s in core_state)
                input=input.float()

                #torch.Size([1, 575]) 2 torch.Size([2, 1, 544]) torch.Size([2, 1, 544]) torch.Size([1, 31]) temp_action
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input.float()
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        nan_mask = torch.isnan(policy_logits)
        num_nans = torch.sum(nan_mask)

        policy_logits=torch.clamp(policy_logits,min=0.0)
        nan_mask = torch.isnan(policy_logits)
        num_nans = torch.sum(nan_mask)
        pickeda = temp_action.view(*policy_logits.shape)
        try:
            if torch.sum(pickeda) > 0:
                row_sum_gt_zero = torch.sum(F.softmax(policy_logits, dim=-1) * pickeda, dim=1) > 0
                if row_sum_gt_zero.all().item():
                    action = F.softmax(policy_logits, dim=-1) * pickeda
                    action = torch.distributions.Categorical(action).sample()
                else:
                    action = F.softmax(policy_logits, dim=-1) * pickeda
                    action = torch.distributions.Categorical(action).sample()
                #action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
            else:

                action = F.softmax(policy_logits, dim=-1)
                action = torch.distributions.Categorical(action).sample()
        except Exception as ex:
            return (
            dict(policy_logits=None, baseline=None, action=None),
            None,
        )

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


Net = AtariNet


def create_env(flags):
    if flags.eval==1:
        env=LTREnvV2(data_path=flags.train_data_path, model_path=flags.mpath,  # data_path=file_path + test_data_path
                       tokenizer_path=flags.mpath, action_space_dim=31, report_count=None, max_len=512,
                       use_gpu=True, caching=True, file_path=flags.file_path, project_list=[flags.project_name], test_env=True,
                       estimate=flags.estimate, metric=flags.metric, non_original=flags.non_original, reg_path=flags.reg_path)
    else:
        env = LTREnvV2(data_path=flags.train_data_path, model_path=flags.mpath,
                       tokenizer_path=flags.mpath, action_space_dim=31, report_count=100, max_len=512,
                       use_gpu=False, caching=True, file_path=flags.file_path, project_list=[flags.project_name], metric=flags.metric,
                       non_original=flags.non_original, reg_path=flags.reg_path)
    return env


def main(flags):
    if flags.eval == 0:
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    current_directory = os.getcwd()

    parser.add_argument("--env", type=str, default="MsPacmanNoFrameskip-v4",
                        help="Gym environment.")
    parser.add_argument("--mode", default="train",
                        choices=["train", "test", "test_render"],
                        help="Training or test mode.")
    parser.add_argument("--xpid", default=None,
                        help="Experiment id (default: None).")

    # Training settings.
    parser.add_argument("--disable_checkpoint", action="store_true",
                        help="Disable saving checkpoint.")
    parser.add_argument("--savedir", default="/scratch/nstevia/100kfinetuning/torchbeastwujivtrace1/logs/torchbeast",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--num_actors", default=0, type=int, metavar="N",
                        help="Number of actors (default: 4).")
    parser.add_argument("--total_steps", default=200000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=4, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--remain", default=0, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--stilltrain", default=0, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--indice", default=1, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument('--eval', default=0, type=int, help='')
    parser.add_argument("--unroll_length", default=1, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num_buffers", default=None, type=int,
                        metavar="N", help="Number of shared-memory buffers.")
    parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                        metavar="N", help="Number learner threads.")
    parser.add_argument("--disable_cuda", default=False, type=bool,
                        help="Disable CUDA.")
    parser.add_argument("--use_lstm", default=True, action="store_true",
                        help="Use LSTM in agent model.")
    parser.add_argument('--cache_path', default=r"C:\Users\phili\Downloads\10428077\Replication\.buffer_cache_ac",
                        help='Cache Path')
    parser.add_argument('--prev_policy_model_path', default=None, help='Trained Policy Path')
    parser.add_argument('--prev_value_model_path', default=None, help='Trained Value Path')
    parser.add_argument('--train_data_path',
                        default=r'/Volumes/Engineering/Harnessing_generalist_agents_on_SE/MGDT_experiments/bug_localization/aspectJ/MAENT/AspectJ_train_for_mining_before.csv',
                        help='Training Data Path')
    parser.add_argument('--save_path',
                        default=r'C:\Users\phili\Downloads\10428077\Replication\LTR\Models\AC\Entropy\AspectJ\\',
                        help='Save Path')
    parser.add_argument('--project_name', default='AspectJ', help='Project Name')
    parser.add_argument('--algo', default='AC_WITHOUT_ENTROPY', type=str, help='Project Name')
    parser.add_argument('--monobeast', default='1', type=str, help='Project Name')
    parser.add_argument('--reg_path', default='1', type=str, help='Project Name')
    parser.add_argument('--mpath', default='1', type=str, help='Project Name')
    parser.add_argument('--file_path', default='1', type=str, help='Project Name') #results_test box_test finalr results_train
    parser.add_argument('--results_test', default='1', type=str, help='Project Name')
    parser.add_argument('--box_test', default='1', type=str, help='Project Name')
    parser.add_argument('--finalr', default='1', type=str, help='Project Name')
    parser.add_argument('--results_train', default='1', type=str, help='Project Name')

    parser.add_argument('--finetune', default='1_f', type=str, help='Project Name')
    parser.add_argument('--non_original', default=0, type=int, help='Project Name')
    parser.add_argument('--metric', default=0, type=int, help='Project Name')
    parser.add_argument('--estimate', default=False, type=bool, help='Project Name')
    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.0006,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--discounting", default=0.99,
                        type=float, help="Discounting factor.")
    parser.add_argument("--reward_clipping", default="none",
                        choices=["abs_one", "none"],
                        help="Reward clipping.")
    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.00048,
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--alpha", default=0.99, type=float,
                        help="RMSProp smoothing constant.")
    parser.add_argument("--momentum", default=0, type=float,
                        help="RMSProp momentum.")
    parser.add_argument("--epsilon", default=0.01, type=float,
                        help="RMSProp epsilon.")
    parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                        help="Global gradient norm clip.")  # 40.0
    flags = parser.parse_args()

    file_path = os.path.join(current_directory,flags.finetune, flags.monobeast+'_LTR')
    cache_path = os.path.join(file_path, '.buffer_cache_ac')
    prev_policy_model_path = flags.prev_policy_model_path
    prev_value_model_path = flags.prev_value_model_path
    train_data_path = flags.train_data_path# r'C:\Users\phili\Downloads\10428077\Replication\LTR\Models\AC\Entropy\AspectJ\\'
    #train_data_path='/scratch/f/foutsekh/nstevia/bug_localization/RL_Model/bug_loc_experiments/AspectJ_train_for_mining_before.csv'
    project_name = flags.project_name
    save_path = os.path.join(file_path, 'Models',flags.algo,flags.project_name)

    #current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,flags.finetune,  flags.algo+'_'+flags.project_name+'_'+flags.monobeast)
    os.makedirs(final_directory, exist_ok=True)


    Path(file_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True,exist_ok=True)
    #used to be data_path=file_path + train_data_path,
    if flags.eval == 1:
        flags.results_test = os.path.join(final_directory, 'results_testn.txt')
        flags.box_test = os.path.join(final_directory, 'box_testn.txt')
        flags.finalr = os.path.join(final_directory, 'finalrn.txt')
        fi = open(flags.finalr, 'a+')
        fi.write('mrr,actualrank' + '\n')
        fi.close()
        fi = open(flags.results_test, 'a+')
        fi.write('algo,project_name,reward,time,episode,steps,mrr,map,actualr,top1,top5,top10' + '\n')
        fi.close()
        fi = open(flags.box_test, 'a+')
        fi.write('algo,project_name,bug_id,mrr,map,top1,top5,top10' + '\n')
        fi.close()
    else:
        flags.results_train = os.path.join(final_directory, 'results_train.txt')
        fi = open(flags.results_train, 'a+')
        fi.write('algo,project_name,reward,time,episode,steps' + '\n')
        fi.close()
    mpath='/scratch/f/foutsekh/nstevia/bug_localization/micro_codebert'
    flags.mpath=mpath
    flags.file_path=file_path
    flags.save_path=save_path

    main(flags)
