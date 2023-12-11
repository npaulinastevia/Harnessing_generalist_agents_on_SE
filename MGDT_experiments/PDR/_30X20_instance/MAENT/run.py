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
import torch.utils.data.distributed
from Params import configs

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
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='garbage_collection_threshold:0.5,max_split_size_mb:128'
import tensorflow as tf
#from wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
 #   VonNeumannMotion
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
if torch.cuda.is_available():
    print('yesssss')
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hide GPU from tf, since tf.io.encode_jpeg/decode_jpeg seem to cause GPU memory leak.
tf.config.set_visible_devices([], "GPU")
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
from JSSP_Env import SJSSP


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
        self.adj_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.fea_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.mask_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.candidate_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.env = env
        self.data=None

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
        #obs = self.env.reset(**kwargs)
        from uniform_instance_gen import uni_instance_gen
        data_generator = uni_instance_gen
        #env = SJSSP(n_j=configs.n_j, n_m=configs.n_m)
        adj, fea, candidate, mask = self.env.reset(
            data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))
        if self.data is not None:
            
            adj, fea, candidate, mask = self.env.reset(self.data)
        #if self.jpeg_obs:
        #    obs = self._process_jpeg(obs)

        # Create a N-1 "done" past frames.
        #self.pad_current_episode(obs, self.num_stack_frames - 1)

        # Create current frame (but with placeholder actions and rewards).

        self.obs_stack.append(0)
        self.act_stack.append(0)
        self.rew_stack.append(0)
        self.done_stack.append(0)
        self.adj_stack=adj
        self.fea_stack=fea
        self.mask_stack=mask
        self.candidate_stack=candidate
        self.info_stack.append(None)
        return self._get_obs()

    def step(self, action: np.ndarray):
        """Replaces env observation with fixed length observation history."""
        # Update applied action to the previous timestep.
        self.act_stack[-1] = action

        adj, fea, reward, done, candidate, mask = self.env.step(action)
        #if self.jpeg_obs:
        #    obs = self._process_jpeg(obs)
        self.rew_stack[-1] = reward
        # Update frame stack.
        self.obs_stack.append(0)
        self.adj_stack=adj
        self.fea_stack=fea
        self.mask_stack=mask
        self.candidate_stack=candidate
        #self.act_stack.append(0)  # Append unknown action to current timestep.
        #self.rew_stack.append(0)
        self.info_stack.append(None)
        return self._get_obs(), reward, done, None

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
        'adj': self.adj_stack,
        'fea': self.fea_stack,
        'candidate': self.candidate_stack,
        'mask': self.mask_stack,
        }
        return episode_history


# from https://github.com/facebookresearch/moolib/blob/06e7a3e80c9f52729b4a6159f3fb4fc78986c98e/examples/atari/environment.py
def create_env(env_name, sticky_actions=False, noop_max=30, terminal_on_life_loss=False):

    # env = gym.wrappers.FrameStack(env, num_stack=4)  # frame stack done separately
    #env = Env()
    #env = SequenceEnvironmentWrapper(env, num_stack_frames=4, jpeg_obs=True)
    from JSSP_Env import SJSSP
    env= SJSSP(n_j=configs.n_j, n_m=configs.n_m)
    env = SequenceEnvironmentWrapper(env, num_stack_frames=1, jpeg_obs=True)
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
            obs = {k: torch.tensor(v) for k, v in obs.items()} #{k: torch.tensor(v).cuda() for k, v in obs.items()}
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


class Experiment:
    def __init__(self, variant,parser):
        args = parser.parse_args()
        ngpus_per_node = torch.cuda.device_count()

        """ This next line is the key to getting DistributedDataParallel working on SLURM:
        		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
         		current process inside a node and is also 0 or 1 in this example."""

        #local_rank = int(os.environ.get("SLURM_LOCALID"))
        #rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

        #current_device = local_rank
        #self.current_device=current_device
        #torch.cuda.set_device(current_device)
        #dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
         #                       rank=rank)
        OBSERVATION_SHAPE = (20, 20) #(84, 84) (20, 20)
        PATCH_SHAPE = (5, 5) # The size of tensor a (16) must match the size of tensor b (36) at non-singleton dimension 2
        NUM_ACTIONS = 1  # 18 Maximum number of actions in the full dataset.
        # rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
        NUM_REWARDS = 3
        RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.target_entropy = -self.act_dim
        self.logger = Logger(variant)
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
            parser=parser,
        )

        # --- Load pretrained weights
        self.parser = parser
        from load_pretrained import load_jax_weights
        model_params, model_state = pickle.load(open("/scratch/nstevia/online-dt4/scripts/checkpoint_38274228.pkl", "rb"))
        self.device = None #variant.get("device", "cuda")
        load_jax_weights(self.model, model_params)
        self.model = self.model # self.model.cuda()
        #for param_tensor in self.model.state_dict():

        #self._load_model_pretrain(path_prefix=self.logger.log_path)

        self.f = open('18bug_log_RELINE_addtrain10K.txt', 'w')
        self.f.close()
        #self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
         #   variant["env"]
        #)
        # initialize by offline trajs
        self.state_mean, self.state_std=None,None
        self.replay_buffer = ReplayBuffer(variant["replay_size"])
        fi = open('18bug_log_addtrain10K.txt', 'a+')
        fi.write('episodes,numberofbugs,mean_reward,loss,time,timeeval'+ os.linesep)
        fi.close()
        fi = open('18typeofbugs_addtrain10K.txt', 'a+')
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

        self.tempera=self.model.temperature()
        self.entro=self.model.target_entropy
        self._load_model_pretrain(path_prefix=self.logger.log_path)


    def _get_env_spec(self, variant):

        #import d4rl_atari
        sticky_actions = False
        noop_max = 30
        terminal_on_life_loss = False
        # env_name=variant["env"]
        # env=create_env(env_name)
        # envT = create_env(env_name)
        # #env = gym.make(variant["env"])
        # #env_to_get_statedim=gym.make(variant["env"])
        # inter=np.array([envT.reset()['observations']])
        # state_dim = inter.shape
        #
        # #state_dim = env_to_get_statedim.reset().flatten().shape[0]
        #
        #
        act_dim = 1

        # env.close()
        return None, act_dim, None

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
        path_prefix = '/scratch/nstevia/online-dt-wuji/multigame-dt/exp/2023.07.24/093003-default'
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
    def _load_model_pretrain(self, path_prefix):
        path_prefix='/scratch/nstevia/100kfinetuning/mgdt3020sac5/exp/2023.10.27/025147-default'
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint_pretrain = torch.load(f)
            for name, target_param in self.model.named_parameters():
                 for param in checkpoint_pretrain["model_state_dict"]:
                     if param==name:
                         target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                                    param].shape == target_param.shape  else print(
                             checkpoint_pretrain["model_state_dict"][param].shape,
                             target_param.shape)
            self.optimizer.load_state_dict(checkpoint_pretrain["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint_pretrain["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint_pretrain["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint_pretrain["pretrain_iter"]
            self.online_iter = checkpoint_pretrain["online_iter"]
            self.total_transitions_sampled = checkpoint_pretrain["total_transitions_sampled"]
            np.random.set_state(checkpoint_pretrain["np"])
            random.setstate(checkpoint_pretrain["python"])
            torch.set_rng_state(checkpoint_pretrain["pytorch"])
        # if Path(f"{path_prefix}/model.pt").exists():
        #     with open(f"{path_prefix}/model.pt", "rb") as f:
        #         checkpoint = torch.load(f)
        #     for param, target_param in zip(checkpoint["model_state_dict"], self.model.parameters()):
        #         target_param.data.copy_(checkpoint["model_state_dict"][param].data) if checkpoint["model_state_dict"][param].shape == target_param.shape else print(checkpoint["model_state_dict"][param].shape,
        #                                                                                             target_param.shape)
            #self.model_pretrain.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model pre loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):
        def _batch_rollout(envs, policy_fn, num_episodes, log_interval=None):
            r"""Roll out a batch of environments under a given policy function."""

            trajectories = []


            num_batch = len(envs)
            #num_steps = envs[0].spec.max_episode_steps
            num_steps = 10000
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
                    obs = {k: torch.tensor(v) for k, v in obs.items()} #{k: torch.tensor(v).cuda() for k, v in obs.items()}
                    o.append(obs['observations'].cpu().numpy()[0])
                    a.append(obs['actions'].cpu().numpy()[0])
                    r.append(obs['rewards'].cpu().numpy()[0])
                    d.append(done_prev[0])
                      # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])
                    actions,_,_,rg = self.model.optimal_action(obs,  return_range=RETURN_RANGE,
            single_return_token=True,
            opt_weight=0,
            num_samples=128,
            action_temperature=1.0,
            return_temperature=0.75,
            action_top_percentile=50,
            return_top_percentile=None,rng=rng, deterministic=False)
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
        #trajectories = [trajectories[ii] for ii in sorted_inds]

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

            trajs,number,r,b,t = vec_evaluate_episode_rtg(  #returns, lengths, trajs = vec_evaluate_episode_rtg(
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
        # for bugstep in b:
        #     if bugstep is not None:
        #         self.buglist.append(bugstep)
        # self.numberOfbugs=self.numberOfbugs+number
        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        return t
        #self.total_transitions_sampled += np.sum(lengths)

        #return {
       #     "aug_traj/return": np.mean(returns),
       #     "aug_traj/length": np.mean(lengths),
       # }



    def evaluate(self): #def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        online_envs=None
        max_ep_len = MAX_EPISODE_LEN
        num_envs=1
        target_explore = self.variant["online_rtg"]
        target_return = [target_explore * self.reward_scale] * num_envs
        trajs,number,r,b,t = vec_evaluate_episode_rtg(  #returns, lengths, trajs = vec_evaluate_episode_rtg(
                create_env,
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                self.buglist,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="eval",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                image_in=True
            )
        self.r=r


        #for eval_fn in eval_fns:
        #    o = eval_fn(self.model)
        #    outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        #eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs,t

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
        evaluate = False
        if evaluate:
            maxon=1
        else:
            maxon = 120000

        while self.online_iter < maxon:#self.variant["max_online_iters"]:

            start = datetime.datetime.now()
            outputs = {}
            if not evaluate:
                x=self._augment_trajectories(

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

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = False
            else:
                evaluation = False
            out = {"time/evaluation": 0}
            if evaluate:
                out,x=self.evaluate()
            self.online_iter = self.online_iter + x
            print('number of bugs is and online iter is', len(self.replay_buffer.trajectories), self.online_iter)
            if not evaluate:
                #if self.online_iter < self.variant["replay_size"]:
                 #   continue
                train_outputs = trainer.train_iteration(
                    loss_fn=loss_fn,
                    dataloader=dataloader,
                    #parser=self.parser,
                    #current_device=self.current_device,
                    #temp=None, #self.model.temperature()
                  #ent=None, #self.model.target_entropy
                #log_temperature_optimizer=None #self.log_temperature_optimizer
                )


            if not evaluate:
                with open("18myfile_loss_addtrain10K.txt", 'a+') as f:
                    for key, value in train_outputs.items():
                        f.write('%s:%s\n' % (key, value))
            fi = open('18bug_log_addtrain10K.txt', 'a+')
            fi.write(str(self.online_iter)+','+str(self.numberOfbugs)+','+str(self.r)+','+str(None)+','+str(str(millis_interval(start, datetime.datetime.now())/1000))+','+str(out["time/evaluation"])+ os.linesep)
            fi.close()
            if not evaluate:
                outputs.update(train_outputs)

            #if evaluation:
             #   eval_outputs, eval_reward = self.evaluate(eval_fns)
              #  outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics

            self.logger.log_metrics(
                outputs,
                 iter_num=self.pretrain_iter + self.online_iter,
                 total_transitions_sampled=self.total_transitions_sampled,
                 writer=writer,
             )
            #
            if not evaluate:
                self._save_model(
                     path_prefix=self.logger.log_path,
                     is_pretrain_model=True,
                 )


        evaluate = True
        self.online_iter=0
        if evaluate:
            maxon=1
        else:
            maxon = 10000
        while self.online_iter < maxon:#self.variant["max_online_iters"]:
            start = datetime.datetime.now()
            outputs = {}
            if not evaluate:
                x=self._augment_trajectories(

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

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = False
            else:
                evaluation = False
            out = {"time/evaluation": 0}
            if evaluate:
                out,x=self.evaluate()
            self.online_iter = self.online_iter + x
            print('number of bugs is and online iter is', len(self.replay_buffer.trajectories), self.online_iter)
            if not evaluate:
                
                train_outputs = trainer.train_iteration(
                    loss_fn=loss_fn,
                    dataloader=dataloader,
                    #parser=self.parser,
                    #current_device=self.current_device,
                    #temp=None, #self.model.temperature()
                  #ent=None, #self.model.target_entropy
                #log_temperature_optimizer=None #self.log_temperature_optimizer
                )


            if not evaluate:
                with open("18myfile_loss_addtrain10K.txt", 'a+') as f:
                    for key, value in train_outputs.items():
                        f.write('%s:%s\n' % (key, value))
            fi = open('18bug_log_addtrain10K_test.txt', 'a+')
            fi.write(str(self.online_iter)+','+str(self.numberOfbugs)+','+str(self.r)+','+str(None)+','+str(str(millis_interval(start, datetime.datetime.now())/1000))+','+str(out["time/evaluation"])+ os.linesep)
            fi.close()
            if not evaluate:
                outputs.update(train_outputs)

            #if evaluation:
             #   eval_outputs, eval_reward = self.evaluate(eval_fns)
              #  outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics

            self.logger.log_metrics(
                outputs,
                 iter_num=self.pretrain_iter + self.online_iter,
                 total_transitions_sampled=self.total_transitions_sampled,
                 writer=writer,
             )
            #
            if not evaluate:
                self._save_model(
                     path_prefix=self.logger.log_path,
                     is_pretrain_model=True,
                 )
            assert False



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
    parser.add_argument("--batch_size", type=int, default=32) #256
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
    parser.add_argument("--replay_size", type=int, default=10000) #
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300) #300
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    #experiment = Experiment(vars(args))
    experiment = Experiment(vars(args), parser)
    print("=" * 50)
    experiment()

