# **********************************************************************************************************************
# **********************************************************************************************************************
# ***                          Using Reinforcement Learning for Load Testing of Video Games                          ***
# ***                                                 Game: MsPacman                                                 ***
# ***                                     RELINE: DQN model + injected bugs info                                     ***
# ***                                       Training for 1000 + 1000 episodes                                        ***
# **********************************************************************************************************************
# **********************************************************************************************************************

import math

import numpy
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
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.utils.data.distributed

import argparse

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
from lib import dqn_model
from lib import wrappers
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import collections
import cv2
import datetime
import gym
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
MAX_EPISODE_LEN = 1000
# --- Create model
from multigame_dt import MultiGameDecisionTransformer

DEFAULT_ENV_NAME = "MsPacmanNoFrameskip-v4"
MEAN_REWARD_BOUND = 400
MAX_ITERATIONS = 1000

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 60#10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

# experience unit : state, action -> new_state, reward, done or not
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state','rtg'])


# Experience "container" with a fixed capacity (i.e. max experiences)
def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis
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
def create_env(env_name, sticky_actions=False, noop_max=30, terminal_on_life_loss=False):

    # env = gym.wrappers.FrameStack(env, num_stack=4)  # frame stack done separately
    env = Env()
    env = SequenceEnvironmentWrapper(env, num_stack_frames=4, jpeg_obs=True)
    return env
def _get_env_spec(variant):
    # import d4rl_atari
    sticky_actions = False
    noop_max = 30
    terminal_on_life_loss = False
    env_name = variant["env"]
    env = create_env(env_name)
    envT = create_env(env_name)
    # env = gym.make(variant["env"])
    # env_to_get_statedim=gym.make(variant["env"])
    inter = np.array([envT.reset()['observations']])
    state_dim = inter.shape

    # state_dim = env_to_get_statedim.reset().flatten().shape[0]

    act_dim = env.action_space.n
    action_range = [
        0,
        env.action_space.n - 1,
    ]
    env.close()
    return state_dim, act_dim, action_range
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    # add experience
    def append(self, experience):
        self.buffer.append(experience)

    # provide a random batch of the experience
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states,rtg = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states),np.array(rtg)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.count_total_moves = 0
        self.count_random_moves = 0


    def _reset(self):
        obs_list = [env.reset()]  # [env.reset(seed=seeds[i]) for i, env in enumerate(envs)]
        self.obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
        self.obs = {k: torch.tensor(v) for k, v in self.obs.items()}
        self.state = self.obs['observations'][0]
        self.traj = {'observations': np.array([]), 'old_observations': np.array([]), 'actions2': np.array([]),
                'rewards': np.array([]),
                'terminals': np.array([]), 'rewards2': np.array([]), 'actions': np.array([]),
                "returns-to-go": np.array([]), }
        self.traj['actions'] = np.append(self.traj['actions'], 0)
        self.traj['rewards'] = np.append(self.traj['rewards'], 0)
        self.o, self.old, self.a, self.r, self.d, self.rtg = [], [], [], [], [], []
        self.total_reward = 0.0
        self.count_total_moves = 0
        self.count_random_moves = 0
        self.rng=rng = torch.Generator()



    @torch.no_grad()  # disable gradient calculation. It will reduce memory consumption.
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        optimal_action_fn = functools.partial(
            net.optimal_action,
            return_range=RETURN_RANGE,
            single_return_token=True,
            opt_weight=0,
            num_samples=128,
            action_temperature=1.0,
            return_temperature=0.75,
            action_top_percentile=50,
            return_top_percentile=None,
        )
        if np.random.random() < epsilon:
            # random action (eps-Greedy)
            # action = env.action_space.sample()
            actions = random.randint(0, 3)
            self.count_random_moves += 1
            self.count_total_moves += 1
            actions=torch.tensor([actions])
            print(self.obs['actions'])
            rtg=numpy.zeros_like(self.obs['actions'])


        else:
            # net action
            actions, _, _, rg = optimal_action_fn(self.obs, rng=self.rng, deterministic=False)
            print(self.obs,'obsddd',rg.shape)
            rtg=rg.cpu().numpy()
            self.count_total_moves += 1

        # do step in the environment

        step_results = [env.step(act) for act in actions.cpu().numpy()]
        obs_list = [result[0] for result in step_results]

        self.obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

        reward = np.stack([result[1] for result in step_results])[0]

        is_done = np.stack([result[2] for result in step_results])[0]


        new_state=self.obs['observations'][0]

        self.total_reward += reward

        # save the experience
        exp = Experience(self.state, actions[0], reward, is_done, new_state,rtg)
        self.exp_buffer.append(exp)
        self.state = new_state  # update the state
        # episode is over
        if is_done:
            done_reward = self.total_reward
            print('tot random moves: %d / %d (%.2f %s) with epsilon: %.2f' % (
                self.count_random_moves, self.count_total_moves,
                (self.count_random_moves * 100 / self.count_total_moves), '%', epsilon))
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states, rtg = batch
    print('ici', states.shape, actions.shape, rewards.shape, dones.shape, next_states.shape, dones)
    assert False
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0  # no discounted reward for done states
    next_state_values = next_state_values.detach()  # return the tensor without connection to its calculation history

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# **********************************************************************************************************************
# *                                                   TRAINING START                                                   *
# **********************************************************************************************************************


if __name__ == "__main__":
    print('\n\n***********************************************************')
    print("* RELINE model's training on MsPacman game is starting... *")
    print('***********************************************************\n')
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
    parser.add_argument("--batch_size", type=int, default=64) #256
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
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300) #300
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    args = parser.parse_args()
    variant=vars(args)
    # set the device -> cuda or cpu
    device = "cpu"

    # create the wrapped environment
    OBSERVATION_SHAPE = (20, 20)  # (84, 84) (20, 20)
    PATCH_SHAPE = (5, 5)  # The size of tensor a (16) must match the size of tensor b (36) at non-singleton dimension 2
    NUM_ACTIONS = 4  # 18 Maximum number of actions in the full dataset.
    # rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
    NUM_REWARDS = 3
    RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset
    state_dim, act_dim, action_range = _get_env_spec(variant)
    target_entropy = -act_dim

    model = MultiGameDecisionTransformer(
        img_size=OBSERVATION_SHAPE,
        patch_size=PATCH_SHAPE,
        num_actions=NUM_ACTIONS,
        num_rewards=NUM_REWARDS,
        return_range=RETURN_RANGE,
        d_model=1280,
        num_layers=10,
        act_dim=act_dim,
        dropout_rate=0.1,
        predict_reward=True,
        single_return_token=True,
        conv_dim=256,
        stochastic_policy=False,
        max_ep_len=MAX_EPISODE_LEN,
        eval_context_length=variant["eval_context_length"],
        init_temperature=variant["init_temperature"],
        state_dim=state_dim,
        target_entropy=target_entropy,

    )
    print('arrive ici')
    tgt_net = MultiGameDecisionTransformer(
        img_size=OBSERVATION_SHAPE,
        patch_size=PATCH_SHAPE,
        num_actions=NUM_ACTIONS,
        num_rewards=NUM_REWARDS,
        return_range=RETURN_RANGE,
        d_model=1280,
        num_layers=10,
        act_dim=act_dim,
        dropout_rate=0.1,
        predict_reward=True,
        single_return_token=True,
        conv_dim=256,
        stochastic_policy=False,
        max_ep_len=MAX_EPISODE_LEN,
        eval_context_length=variant["eval_context_length"],
        init_temperature=variant["init_temperature"],
        state_dim=state_dim,
        target_entropy=target_entropy,

    )
    # --- Load pretrained weights

    parser = parser
    from load_pretrained import load_jax_weights

    model_params, model_state = pickle.load(open("./scripts/checkpoint_38274228.pkl", "rb"))
    device = None  # variant.get("device", "cuda")
    load_jax_weights(model, model_params)
    load_jax_weights(tgt_net, model_params)

    f = open('bug_log_RELINE.txt', 'w')
    f.close()
    fi = open('bug_log.txt', 'a+')
    fi.write('episodes,numberofbugs,mean_reward,loss,time,timeeval' + os.linesep)
    fi.close()
    fi = open('typeofbugs.txt', 'a+')
    fi.write('bug,reward' + os.linesep)
    fi.close()
    aug_trajs = []
    r = None

    numberOfbugs = 0
    target_entropy = -act_dim
    buglist = []

    num_actions = 5  # exclude actions: 5 6 7 8
    # 0 -> none
    # 1 -> up
    # 2 -> right
    # 3 -> left
    # 4 -> down

    buffer = ExperienceBuffer(REPLAY_SIZE)
    env=create_env('')
    agent = Agent(create_env(''), buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    f = open('bug_log_RELINE.txt', 'w')
    f.close()
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(model, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = float(np.mean(total_rewards[-100:]))

            e = datetime.datetime.now()
            print("frames: %d, episodes: %d , mean reward: %.3f, eps: %.2f, speed: %.2f f/s, time: %s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed, e.strftime("%Y-%m-%d %H:%M:%S")))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(model.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if len(total_rewards) == MAX_ITERATIONS:
                print("training ends")
                break

        # not enough experience for the training
        if len(buffer) < REPLAY_START_SIZE:
            continue
        # update target net
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), DEFAULT_ENV_NAME + "-last.dat")
            print('Target net update at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx == REPLAY_SIZE:
            print('Experience replay buffer full at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx % REPLAY_SIZE == 0 and frame_idx > REPLAY_SIZE:
            print('Experience replay buffer refilled with new experiences at frame: %d , games: %d'
                  % (frame_idx, len(total_rewards)))

        if frame_idx == REPLAY_START_SIZE:
            print('Training starts at frame: %d , games: %d' % (frame_idx, len(total_rewards)))

        if frame_idx == EPSILON_DECAY_LAST_FRAME:
            print('Epsilon reaches the minimum value at frame: %d , games: %d' % (frame_idx, len(total_rewards)))
            tgt_net.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), DEFAULT_ENV_NAME + "-last.dat")

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, model, tgt_net, device=device)
        # print('loss: %.3f , frame: %d , games: %d' % (loss_t, frame_idx, len(total_rewards)))
        loss_t.backward()
        optimizer.step()

    env.close()

    lines = [line for line in open('bug_log_RELINE.txt', 'r')]
    lines_1k = lines[-1000:]

    count_0bug = 0
    count_1bug = 0
    count_2bug = 0
    count_3bug = 0
    count_4bug = 0



    print('\nReport injected bugs spotted during last 1000 episodes:')
    print('0 injected bug spotted in %d episodes' % count_0bug)
    print('1 injected bug spotted in %d episodes' % count_1bug)
    print('2 injected bugs spotted in %d episodes' % count_2bug)
    print('3 injected bugs spotted in %d episodes' % count_3bug)
    print('4 injected bugs spotted in %d episodes' % count_4bug)
    print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")

#                                                                                                               \    /\
#                                                                                                                )  ( ')
#                                                                                                               (  /  )
#                                                                                                                \(__)|
