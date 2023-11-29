"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import functools
import os
import random
from PIL import Image
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

import collections

# import cv2
MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
        vec_env,
        eval_rtg,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        device,
        use_mean=False,
        reward_scale=0.001,
):
    def eval_episodes_fn(model):
        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        returns, lengths, _ = vec_evaluate_episode_rtg(
            vec_env,
            state_dim,
            act_dim,
            model,
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
        )
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg_or(

        vec_env,
        state_dim,
        act_dim,
        model,
        target_return: list,
        max_ep_len=1000,
        reward_scale=0.001,
        state_mean=0.0,
        state_std=1.0,
        device="cuda",
        mode="normal",
        use_mean=False,
        image_in=True,
):
    assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"

    if image_in:
        states = (
            torch.from_numpy(state)
                .reshape(num_envs, state_dim[0], state_dim[1])
                .to(device=device, dtype=torch.float32)
        ).reshape(num_envs, -1, state_dim[0], state_dim[1])
    else:
        states = (
            torch.from_numpy(state)
                .reshape(num_envs, state_dim)
                .to(device=device, dtype=torch.float32)
        ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

        # the return action is a SquashNormal distribution
        action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
        if use_mean:
            action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim[0], state_dim[1])
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )


@torch.no_grad()
def vec_evaluate_episode_rtg(
        create_env,
        vec_env,
        state_dim,
        act_dim,
        model,
        buglist,
        target_return: list,
        max_ep_len=1000,
        reward_scale=0.001,
        state_mean=0.0,
        state_std=1.0,
        device="cuda",
        mode="normal",
        use_mean=False,
        image_in=True,
):
    # assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)
    numberbugs = 0
    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

    # num_envs = vec_env.num_envs
    # state = vec_env.reset()
    RETURN_RANGE = [-20, 100]
    optimal_action_fn = functools.partial(
        model.optimal_action,
        return_range=RETURN_RANGE,
        single_return_token=True,
        opt_weight=0,
        num_samples=128,
        action_temperature=1.0,
        return_temperature=0.75,
        action_top_percentile=50,
        return_top_percentile=None,
    )
    env_name = "MsPacman"
    num_envs = 1
    env_fn = lambda: create_env(env_name)

    envs = [env_fn() for _ in range(num_envs)]
    num_episodes = 1
    lebug = []
    trajectories = []

    num_batch = len(envs)
    # num_steps = envs[0].spec.max_episode_steps

    num_steps = 2
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_episodes)]

    rew_sum_list = []
    for c in range(1):  # num_episodes // num_batch
        o, a, r, d, rtg = [], [], [], [], []
        seeds = seeds_list[c * num_batch: (c + 1) * num_batch]
        rng.manual_seed(seeds[0])
        bug_flags = [False, False, False, False]
        obs_list = [env.reset() for i, env in enumerate(envs)]

        obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

        done = np.zeros(num_batch, dtype=np.int32)
        traj = {'observations': np.array([]), 'actions2': np.array([]), 'rewards': np.array([]),
                'terminals': np.array([]), 'rewards2': np.array([]), 'actions': np.array([]),
                "returns-to-go": np.array([]), }
        traj['actions'] = np.append(traj['actions'], 0)
        traj['rewards'] = np.append(traj['rewards'], 0)
        total_rew = 0
        t = 0
        while True:  # for t in range(num_steps)

            done_prev = done
            obs = {k: torch.tensor(v, device=device) for k, v in obs.items()}
            o.append(obs['observations'].cpu().numpy()[0])
            a.append(obs['actions'].cpu().numpy()[0])
            r.append(obs['rewards'].cpu().numpy()[0])
            d.append(done_prev[0])
            # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])
            actions, _, _, rg = optimal_action_fn(obs, rng=rng, deterministic=False)
            rtg.append(rg.cpu().numpy()[0])
            # Collect step results and stack as a batch.
            step_results = [env.step(act) for env, act in zip(envs, actions.cpu().numpy())]

            obs_list = [result[0] for result in step_results]
            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
            rew = np.stack([result[1] for result in step_results])

            if step_results[0][3]['bug'] is not None:
                if step_results[0][3]['bug'] not in buglist and step_results[0][3]['bug'] not in lebug:
                    numberbugs += 1
                    lebug.append(step_results[0][3]['bug'])
                    fi = open('typeofbugs.txt', 'a+')
                    fi.write(str(step_results[0][3]['bug']) + ',' + str(rew[0]) + os.linesep)
                    fi.close()

            done = np.stack([result[2] for result in step_results])
            traj['actions'] = np.append(traj['actions'], actions.cpu().numpy()[0])

            done = np.logical_or(done, done_prev).astype(np.int32)
            rew = rew * (1 - done)
            total_rew += rew
            traj['rewards'] = np.append(traj['rewards'], rew)
            t = t + 1
            if np.all(done) or t == num_steps - 1:
                o.append(obs['observations'][0])
                a.append(obs['actions'][0])
                r.append(obs['rewards'][0])
                rtg.append(torch.zeros_like(rg).cpu().numpy()[0])
                d.append(1)

                traj['actions2'] = np.stack((p for p in a), axis=0)

                traj['observations'] = np.stack((p for p in o), axis=0)

                traj['rewards2'] = np.stack((p for p in r), axis=0)
                traj["returns-to-go"] = np.stack((p for p in rtg), axis=0)
                traj['terminals'] = np.array(d).astype(bool)
                trajectories.append(traj)

                break

    return trajectories, numberbugs, total_rew, lebug

    # return (
    #   episode_return.reshape(num_envs),
    #   episode_length.reshape(num_envs),
    #   trajectories,
# )





import math
import torch
import datetime
import torch.nn as nn
import torch
from skimage.draw import random_shapes
from gym.spaces import Box, Discrete
import gc
gc.collect()
torch.cuda.empty_cache()
import collections
import functools
import json
import os
from torch import distributions as pyd
import random
import time
from torch.utils.tensorboard import SummaryWriter

from data import create_dataloader
from evaluation import create_vec_eval_episodes_fn
from trainer import SequenceTrainer
import gym
import numpy as np
import scipy
import torch
import argparse
import pickle
import random
import time
import gym
import torch
import numpy as np
import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
#from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='garbage_collection_threshold:0.5,max_split_size_mb:128'
import tensorflow as tf
from wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
    VonNeumannMotion
gym.logger.set_level(gym.logger.ERROR)

from atari_data import get_human_normalized_score
from atari_preprocessing import AtariPreprocessing

# --- Setup
seed = 100
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hide GPU from tf, since tf.io.encode_jpeg/decode_jpeg seem to cause GPU memory leak.
tf.config.set_visible_devices([], "GPU")

# --- Create environments
def get_maze():
    size = (20, 20)
    max_shapes = 50
    min_shapes = max_shapes // 2
    max_size = 3
    seed = 2
    x, _ = random_shapes(size, max_shapes, min_shapes, max_size=max_size, multichannel=False, random_seed=seed)

    x[x == 255] = 0
    x[np.nonzero(x)] = 1

    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1

    return x


map = get_maze()
start_idx = [[10, 7]]
goal_idx = [[12, 12]]


class Maze(BaseMaze):
    @property
    def size(self):
        return map.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(map == 0), axis=1))
        obstacle = Object('obstacle', 85, color.obstacle, True, np.stack(np.where(map == 1), axis=1))
        agent = Object('agent', 170, color.agent, False, [])
        goal = Object('goal', 255, color.goal, False, [])
        return free, obstacle, agent, goal


class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        self.maze = Maze()
        self.motions = VonNeumannMotion()
        # self.bugs = [
        #     [1,1],[3,4],[7,5],[18,1],[11,12],[18,14],
        #     [12,6],[18,6],[11,14],[1,13],[3,13],[1,17],
        #     [2,18],[10,18],[17,18],[12,18],[15,17]
        # ]
        # self.bugs = np.logical_and(np.random.randint(0,2,[20,20]), np.logical_not(map))
        # self.bugs_cnt = np.count_nonzero(self.bugs)
        #self.bug_idxs = [[0, 1], [3, 4], [1, 6], [7, 5], [6, 17], [5, 11], [7, 1], [0, 10], [16, 10], [18, 1], [4, 1],
         #                [11, 12], [18, 14], [12, 6], [18, 6], [11, 14], [1, 13], [3, 13], [1, 17], [2, 18], [10, 18],
         #                [15, 3], [17, 18], [12, 18], [15, 17]]
        self.bug_idxs = [[1, 2], [1, 6], [1, 7], [1, 8], [1, 17], [2, 2], [2, 3], [2, 7], [2, 9], [2, 10], [2, 11],
                          [2, 12], [2, 18], [3, 1], [3, 3], [3, 7], [3, 8], [3, 9], [3, 11], [3, 14], [3, 15], [3, 16],
                          [3, 17], [3, 18], [4, 1], [4, 2], [4, 3], [4, 7], [4, 9], [4, 11], [4, 14], [4, 17], [5, 4],
                          [5, 5], [5, 8], [5, 11], [5, 16], [6, 5], [6, 8], [7, 1], [7, 3], [7, 4], [7, 7], [7, 8],
                          [7, 9], [7, 11], [7, 17], [7, 18], [8, 1], [8, 2], [8, 8], [8, 9], [8, 11], [8, 12], [8, 13],
                          [8, 18], [9, 2], [9, 10], [9, 11], [9, 13], [9, 14], [10, 2], [10, 4], [10, 9], [10, 15],
                          [10, 16], [11, 10], [11, 11], [12, 1], [12, 4], [12, 5], [12, 6], [12, 11], [12, 12], [12, 13],
                          [13, 3], [13, 5], [13, 6], [13, 10], [13, 11], [14, 4], [14, 5], [14, 6], [14, 8], [14, 9],
                          [14, 10], [14, 16], [14, 17], [14, 18], [15, 1], [15, 2], [15, 5], [15, 6], [15, 7], [15, 8],
                          [15, 9], [15, 15], [16, 10], [17, 1], [17, 5], [17, 6], [17, 7], [17, 8], [17, 14], [17, 15],
                          [17, 17], [17, 18], [18, 1], [18, 2], [18, 3], [18, 4], [18, 6], [18, 7], [18, 12], [18,14]]
        self.bug_cnt = len(self.bug_idxs)
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        self.context = dict(
            inputs=1,
            outputs=self.action_space.n
        )

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        # mark bug position
        bug = tuple(new_position) if new_position in self.bug_idxs else None
        # if bug is not None:
        #     print(bug)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        goal = self._is_goal(new_position)
        if goal:
            reward = +10
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self.maze.to_value(), reward, done, dict(bug=bug, valid=valid, goal=goal)
        #return self.maze.to_value()[..., np.newaxis], reward, done, dict(bug=bug, valid=valid, goal=goal)
    def reset(self):
        self.bug_item = set()
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        return self.maze.to_value()
        #return self.maze.to_value()[..., np.newaxis]

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis
class SequenceEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, num_stack_frames: int = 1, jpeg_obs: bool = False):
        super().__init__(env)
        self.num_stack_frames = num_stack_frames
        self.jpeg_obs = jpeg_obs

        self.obs_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.act_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.rew_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.done_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.info_stack = collections.deque([], maxlen=self.num_stack_frames)

    @property
    def observation_space(self):
        parent_obs_space = self.env.observation_space
        act_space = self.env.action_space
        episode_history = {
            "observations": gym.spaces.Box(
                np.stack([parent_obs_space.low] * self.num_stack_frames, axis=0),
                np.stack([parent_obs_space.high] * self.num_stack_frames, axis=0),
                dtype=parent_obs_space.dtype,
            ),
            "actions": gym.spaces.Box(0, act_space.n, [self.num_stack_frames], dtype=act_space.dtype),
            "rewards": gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames]),
        }
        return gym.spaces.Dict(**episode_history)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self.jpeg_obs:
            obs = self._process_jpeg(obs)

        # Create a N-1 "done" past frames.
        self.pad_current_episode(obs, self.num_stack_frames - 1)
        # Create current frame (but with placeholder actions and rewards).
        self.obs_stack.append(obs)
        self.act_stack.append(0)
        self.rew_stack.append(0)
        self.done_stack.append(0)
        self.info_stack.append(None)
        return self._get_obs()

    def step(self, action: np.ndarray):
        """Replaces env observation with fixed length observation history."""
        # Update applied action to the previous timestep.
        self.act_stack[-1] = action
        obs, rew, done, info = self.env.step(action)
        if self.jpeg_obs:
            obs = self._process_jpeg(obs)
        self.rew_stack[-1] = rew
        # Update frame stack.
        self.obs_stack.append(obs)
        self.act_stack.append(0)  # Append unknown action to current timestep.
        self.rew_stack.append(0)
        self.info_stack.append(info)
        return self._get_obs(), rew, done, info

    def pad_current_episode(self, obs, n):
        # Prepad current episode with n steps.
        for _ in range(n):
            self.obs_stack.append(np.zeros_like(obs))
            self.act_stack.append(0)
            self.rew_stack.append(0)
            self.done_stack.append(1)
            self.info_stack.append(None)

    def _process_jpeg(self, obs):
        obs = np.expand_dims(obs, axis=-1)  # tf expects channel-last
        obs = tf.io.decode_jpeg(tf.io.encode_jpeg(obs))
        obs = np.array(obs).transpose(2, 0, 1)  # to channel-first
        return obs

    def _get_obs(self):
        r"""Return current episode's N-stacked observation.

        For N=3, the first observation of the episode (reset) looks like:

        *= hasn't happened yet.

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    0    0.   0.   True
        g0    x0   0.   0.   False

        After the first step(a0) taken, yielding x1, r0, done0, info0, the next
        observation looks like:

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    x0   0.   0.   False
        g1    x1   a0   r0   d0

        A more chronologically intuitive way to re-order the column data would be:

        PREV_ACT  PREV_REW  PREV_DONE CURR_GOAL CURR_OBS
        ================================================
        0.        0.        True      g0        0
        0.        0.        False*    g0        x0
        a0        r0        info0     g1        x1

        Returns:
        episode_history: np.ndarray of observation.
        """
        episode_history = {
            "observations": np.stack(self.obs_stack, axis=0),
            "actions": np.stack(self.act_stack, axis=0),
            "rewards": np.stack(self.rew_stack, axis=0),
        }
        return episode_history


# from https://github.com/facebookresearch/moolib/blob/06e7a3e80c9f52729b4a6159f3fb4fc78986c98e/examples/atari/environment.py
def create_env(env_name, sticky_actions=False, noop_max=30, terminal_on_life_loss=False):

    # env = gym.wrappers.FrameStack(env, num_stack=4)  # frame stack done separately
    env = Env()
    env = SequenceEnvironmentWrapper(env, num_stack_frames=4, jpeg_obs=True)
    return env


# env_name = "Breakout"
# num_envs = 8
# env_fn = lambda: create_env(env_name)
# envs = [env_fn() for _ in range(num_envs)]
# print(f"num_envs: {num_envs}", envs[0])
import torch

# --- Create offline RL dataset
MAX_EPISODE_LEN = 1000
# --- Create model
from multigame_dt import MultiGameDecisionTransformer



# --- Train model
#model.train()

# --- Save/Load model weights
# torch.save(model.state_dict(), "model.pth")
# model.load_state_dict(torch.load("model.pth"))
MAX_EPISODE_LEN = 1000
# --- Evaluate model
def _batch_rollout1(envs, policy_fn, num_episodes, log_interval=None):
    r"""Roll out a batch of environments under a given policy function."""
    num_batch = len(envs)
    num_steps = envs[0].spec.max_episode_steps
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2**32 - 1) for _ in range(num_episodes)]
    print(f"seeds: {seeds_list}")

    rew_sum_list = []
    for c in range(num_episodes // num_batch):
        seeds = seeds_list[c * num_batch : (c + 1) * num_batch]
        rng.manual_seed(seeds[0])

        obs_list = [env.reset(seed=seeds[i]) for i, env in enumerate(envs)]
        obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
        rew_sum = np.zeros(num_batch, dtype=np.float32)
        done = np.zeros(num_batch, dtype=np.int32)
        start = time.perf_counter()
        for t in range(num_steps):
            done_prev = done
            obs = {k: torch.tensor(v, device=device) for k, v in obs.items()}
            actions = policy_fn(obs, rng=rng, deterministic=False)

            # Collect step results and stack as a batch.
            step_results = [env.step(act) for env, act in zip(envs, actions.cpu().numpy())]
            obs_list = [result[0] for result in step_results]
            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
            rew = np.stack([result[1] for result in step_results])
            done = np.stack([result[2] for result in step_results])

            done = np.logical_or(done, done_prev).astype(np.int32)
            rew = rew * (1 - done)
            rew_sum += rew

            if log_interval and t % log_interval == 0:
                elapsed = time.perf_counter() - start
                print(f"step: {t}, fps: {(num_batch * t / elapsed):.2f}, done: {done.astype(np.int32)}, rew_sum: {rew_sum}")

            # Don't continue if all environments are done.
            if np.all(done):
                break

        rew_sum_list.append(rew_sum)
    return np.concatenate(rew_sum_list)


from replay_buffer import ReplayBuffer
from lib import dqn_model
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
class Experiment:
    def __init__(self, variant):
        OBSERVATION_SHAPE = (20, 20) #(84, 84) (20, 20)
        PATCH_SHAPE = (5, 5) # The size of tensor a (16) must match the size of tensor b (36) at non-singleton dimension 2
        NUM_ACTIONS = 4  # 18 Maximum number of actions in the full dataset.
        # rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
        NUM_REWARDS = 3
        RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.target_entropy = -self.act_dim
        self.epsilon = EPSILON_START
        self.model = MultiGameDecisionTransformer(
            img_size=OBSERVATION_SHAPE,
            patch_size=PATCH_SHAPE,
            num_actions=NUM_ACTIONS,
            num_rewards=NUM_REWARDS,
            return_range=RETURN_RANGE,
            d_model=1280,
            num_layers=10,
            act_dim=self.act_dim,
            dropout_rate=0.1,
            predict_reward=True,
            single_return_token=True,
            conv_dim=256,
            stochastic_policy=False,
            max_ep_len=MAX_EPISODE_LEN,
            eval_context_length=variant["eval_context_length"],
            init_temperature=variant["init_temperature"],
            state_dim=self.state_dim,
            target_entropy=self.target_entropy,
        )

        # --- Load pretrained weights
        from load_pretrained import load_jax_weights
        model_params, model_state = pickle.load(open("./scripts/checkpoint_38274228.pkl", "rb"))
        self.device = variant.get("device", "cuda")
        load_jax_weights(self.model, model_params)
        self.model = self.model.to(device=self.device)
        #self.net = dqn_model.DQN(OBSERVATION_SHAPE, NUM_ACTIONS).to(device=self.device)
        #self.tgt_net = dqn_model.DQN(OBSERVATION_SHAPE, NUM_ACTIONS).to(device=self.device)
        self.f = open('bug_log_RELINE.txt', 'w')
        self.f.close()
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
        fi = open('bug_log.txt', 'a+')
        fi.write('episodes,mean_reward,loss,time'+ os.linesep)
        fi.close()
        fi = open('typeofbugs.txt', 'a+')
        fi.write('bug,reward'+ os.linesep)
        fi.close()
        self.aug_trajs = []
        self.r=None

        self.numberOfbugs=0
        self.target_entropy = -self.act_dim
        self.buglist=[]



        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):

        #import d4rl_atari
        sticky_actions = False
        noop_max = 30
        terminal_on_life_loss = False
        env_name=variant["env"]
        env=create_env(env_name)
        envT = create_env(env_name)
        #env = gym.make(variant["env"])
        #env_to_get_statedim=gym.make(variant["env"])
        inter=np.array([envT.reset()['observations']])
        state_dim = inter.shape

        #state_dim = env_to_get_statedim.reset().flatten().shape[0]


        act_dim = env.action_space.n
        action_range = [
            0,
            env.action_space.n-1,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        torch.save(self.model.state_dict(), "model_as_mgdt.pth")
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):
        def _batch_rollout(envs, policy_fn, num_episodes, log_interval=None):
            r"""Roll out a batch of environments under a given policy function."""

            trajectories = []


            num_batch = len(envs)
            #num_steps = envs[0].spec.max_episode_steps
            num_steps = 2 #27000
            assert num_episodes % num_batch == 0

            rng = torch.Generator()
            seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_episodes)]

            rew_sum_list = []

            for c in range(num_episodes // num_batch):
                o, a, r, d,rtg = [], [], [], [],[]
                seeds = seeds_list[c * num_batch: (c + 1) * num_batch]
                rng.manual_seed(seeds[0])

                obs_list = [env.reset() for i, env in enumerate(envs)] #[env.reset(seed=seeds[i]) for i, env in enumerate(envs)]
                obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

                done = np.zeros(num_batch, dtype=np.int32)
                traj = {'observations': np.array([]), 'actions2': np.array([]), 'rewards': np.array([]),
                        'terminals': np.array([]),'rewards2': np.array([]),'actions': np.array([]),"returns-to-go": np.array([]),}
                traj['actions'] = np.append(traj['actions'], 0)
                traj['rewards'] = np.append(traj['rewards'], 0)

                for t in range(num_steps):
                    done_prev = done
                    obs = {k: torch.tensor(v, device=device) for k, v in obs.items()}
                    o.append(obs['observations'].cpu().numpy()[0])
                    a.append(obs['actions'].cpu().numpy()[0])
                    r.append(obs['rewards'].cpu().numpy()[0])
                    d.append(done_prev[0])
                      # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])
                    actions,_,_,rg = policy_fn(obs, rng=rng, deterministic=False)
                    rtg.append(rg.cpu().numpy()[0])
                    # Collect step results and stack as a batch.

                    step_results = [env.step(act) for env, act in zip(envs, actions.cpu().numpy())]

                    obs_list = [result[0] for result in step_results]
                    obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

                    rew = np.stack([result[1] for result in step_results])

                    done = np.stack([result[2] for result in step_results])
                    traj['actions'] = np.append(traj['actions'], actions.cpu().numpy()[0])

                    done = np.logical_or(done, done_prev).astype(np.int32)
                    rew = rew * (1 - done)

                    traj['rewards'] = np.append(traj['rewards'], rew)
                    if np.all(done) or t==num_steps-1:

                        o.append(obs['observations'][0])
                        a.append(obs['actions'][0])
                        r.append(obs['rewards'][0])
                        rtg.append(torch.zeros_like(rg).cpu().numpy()[0])
                        d.append(1)

                        traj['actions2'] = np.stack((p for p in a),axis=0)

                        traj['observations']=np.stack((p for p in o),axis=0)

                        traj['rewards2'] = np.stack((p for p in r),axis=0)
                        traj["returns-to-go"]=np.stack((p for p in rtg),axis=0)
                        traj['terminals'] = np.array(d).astype(bool)
                        trajectories.append(traj)
                        print(traj['actions2'].shape, traj['observations'].shape,traj['rewards'].shape,traj['terminals'].shape,
                              traj['actions'].shape,traj['rewards2'].shape,traj["returns-to-go"].shape)
                        break

            return trajectories
        self.model.eval()
        RETURN_RANGE = [-20, 100]

        optimal_action_fn = functools.partial(
            self.model.optimal_action,
            return_range=RETURN_RANGE,
            single_return_token=True,
            opt_weight=0,
            num_samples=128,
            action_temperature=1.0,
            return_temperature=0.75,
            action_top_percentile=50,
            return_top_percentile=None,
        )
        env_name = "MsPacman"
        num_envs = 1
        env_fn = lambda: create_env(env_name)

        envs = [env_fn() for _ in range(num_envs)]
        trajectories=_batch_rollout(envs, optimal_action_fn, num_episodes=1, log_interval=100)
        states, traj_lens, returns = [], [], []

        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)
        # used for input normalization
        print(states,'states')
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)
        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN
        num_envs=1
        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * num_envs #[target_explore * self.reward_scale] * online_envs.num_envs

            trajs,number,r,b = vec_evaluate_episode_rtg(  #returns, lengths, trajs = vec_evaluate_episode_rtg(
                create_env,
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                self.buglist,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                image_in=True
            )
        self.r=r
        for bugstep in b:
            if bugstep is not None:
                self.buglist.append(bugstep)
        self.numberOfbugs=self.numberOfbugs+number
        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        #self.total_transitions_sampled += np.sum(lengths)

        #return {
       #     "aug_traj/return": np.mean(returns),
       #     "aug_traj/length": np.mean(lengths),
       # }



    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        #eval_fns = [
        #    create_vec_eval_episodes_fn(
         #       vec_env=eval_envs,
          #      eval_rtg=self.variant["eval_rtg"],
          #      state_dim=self.state_dim,
          #      act_dim=self.act_dim,
           #     state_mean=self.state_mean,
            #    state_std=self.state_std,
           #     device=self.device,
           #     use_mean=True,
           #     reward_scale=self.reward_scale,
          #  )
        #]
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.online_iter < self.variant["max_online_iters"]:
            start = datetime.datetime.now()
            outputs = {}
            self._augment_trajectories(

                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
           # outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )
            print('number of bugs is and online iter is', self.numberOfbugs,self.online_iter)
            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = False
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            with open("myfile_loss.txt", 'a+') as f:
                for key, value in train_outputs.items():
                    f.write('%s:%s\n' % (key, value))
            fi = open('bug_log.txt', 'a+')
            fi.write(str(self.online_iter)+','+str(float(self.r))+','+str(None)+','+str(str(millis_interval(start, datetime.datetime.now())/1000))+ os.linesep)
            fi.close()
            #outputs.update(train_outputs)

            #if evaluation:
             #   eval_outputs, eval_reward = self.evaluate(eval_fns)
              #  outputs.update(eval_outputs)

            #outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            #self.logger.log_metrics(
            #    outputs,
            #     iter_num=self.pretrain_iter + self.online_iter,
            #     total_transitions_sampled=self.total_transitions_sampled,
            #     writer=writer,
            # )
            #
            # self._save_model(
            #     path_prefix=self.logger.log_path,
            #     is_pretrain_model=False,
            # )

            self.online_iter += 1

    def __call__(self):

        utils.set_seed_everywhere(args.seed)



        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():

                import d4rl_atari
                #env = gym.make(env_name)
                env = create_env(env_name)
                env.seed(seed)
                if hasattr(env.env, "wrapped_env"):
                    env.env.wrapped_env.seed(seed)
                elif hasattr(env.env, "seed"):
                    env.env.seed(seed)
                else:
                    pass
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

                if target_goal:
                    env.set_target_goal(target_goal)
                    print(f"Set the target goal to be {env.target_goal}")

                return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None

        #eval_envs = SubprocVecEnv(
         #   [
          #      get_env_builder(i, env_name=env_name, target_goal=target_goal)
          #      for i in range(self.variant["num_eval_episodes"])
         #   ]
       # )
        eval_envs = None

        self.start_time = time.time()

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            #online_envs = SubprocVecEnv(
            #    [
            #        get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
            #        for i in range(self.variant["num_online_rollouts"])
             #   ]
            #)
            online_envs = None

            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default='MsPacman') #hopper-medium-v2

    # model options
    parser.add_argument("--K", type=int, default=20) #20
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256) #256
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1000)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=250) #
    parser.add_argument("--num_updates_per_online_iter", type=int, default=1) #300
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
