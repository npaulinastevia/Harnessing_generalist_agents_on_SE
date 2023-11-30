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
import threading
import time
import timeit
import traceback
import typing
from torchbeast.wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
    VonNeumannMotion
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

from skimage.metrics import structural_similarity as ssim
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
import cv2
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace
from skimage.draw import random_shapes
from gym.spaces import Box, Discrete
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

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
parser.add_argument("--savedir", default="/scratch/nstevia/torchbeastppowuji/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=48, type=int, metavar="N",
                    help="Number of actors (default: 4).")#
parser.add_argument("--K_epochs", default=15, type=int, metavar="N",

                    help="number of opt step for ppo.")#
parser.add_argument("--eps_clip", default=0.2, type=float, metavar="T",
                    help="eps clip for ppo.")
parser.add_argument("--total_steps", default=20000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--gamma", default=0.99, type=float, metavar="T",
                    help="gamma for ppo.")
parser.add_argument("--batch_size", default=32, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")#
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", default=True,action="store_true",
                    help="Use LSTM in agent model.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006, #0.0006
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
parser.add_argument("--tmpdir", default='',
                    type=str,  help="slum tmp dir.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]
def get_maze():
    size = (20, 20)
    max_shapes = 50
    min_shapes = max_shapes // 2
    max_size = 3
    seed = 2
    x, _ = random_shapes(size, max_shapes, min_shapes, max_size=max_size, multichannel=False, random_seed=seed)

    #x[x == 255] = 0
    #x[np.nonzero(x)] = 1

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
class EnvM(BaseEnv):
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
        self.bug_idxs = [[0, 1], [3, 4], [1, 6], [7, 5], [6, 17], [5, 11], [7, 1], [0, 10], [16, 10], [18, 1], [4, 1],
                         [11, 12], [18, 14], [12, 6], [18, 6], [11, 14], [1, 13], [3, 13], [1, 17], [2, 18], [10, 18],
                         [15, 3], [17, 18], [12, 18], [15, 17]]
        # self.bug_idxs = [[1, 2], [1, 6], [1, 7], [1, 8], [1, 17], [2, 2], [2, 3], [2, 7], [2, 9], [2, 10], [2, 11],
        #                   [2, 12], [2, 18], [3, 1], [3, 3], [3, 7], [3, 8], [3, 9], [3, 11], [3, 14], [3, 15], [3, 16],
        #                   [3, 17], [3, 18], [4, 1], [4, 2], [4, 3], [4, 7], [4, 9], [4, 11], [4, 14], [4, 17], [5, 4],
        #                   [5, 5], [5, 8], [5, 11], [5, 16], [6, 5], [6, 8], [7, 1], [7, 3], [7, 4], [7, 7], [7, 8],
        #                   [7, 9], [7, 11], [7, 17], [7, 18], [8, 1], [8, 2], [8, 8], [8, 9], [8, 11], [8, 12], [8, 13],
        #                   [8, 18], [9, 2], [9, 10], [9, 11], [9, 13], [9, 14], [10, 2], [10, 4], [10, 9], [10, 15],
        #                   [10, 16], [11, 10], [11, 11], [12, 1], [12, 4], [12, 5], [12, 6], [12, 11], [12, 12], [12, 13],
        #                   [13, 3], [13, 5], [13, 6], [13, 10], [13, 11], [14, 4], [14, 5], [14, 6], [14, 8], [14, 9],
        #                   [14, 10], [14, 16], [14, 17], [14, 18], [15, 1], [15, 2], [15, 5], [15, 6], [15, 7], [15, 8],
        #                   [15, 9], [15, 15], [16, 10], [17, 1], [17, 5], [17, 6], [17, 7], [17, 8], [17, 14], [17, 15],
        #                   [17, 17], [17, 18], [18, 1], [18, 2], [18, 3], [18, 4], [18, 6], [18, 7], [18, 12], [18,14]]
        self.bug_cnt = len(self.bug_idxs)
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=(20,20,1), dtype=np.uint8) #Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
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
        #return self.maze.to_value(), reward, done, dict(bug=bug, valid=valid, goal=goal)
        return self.maze.to_value()[..., np.newaxis], reward, done, dict(bug=bug, valid=valid, goal=goal)
    def reset(self):
        self.bug_item = set()
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        #return self.maze.to_value()
        return self.maze.to_value()[..., np.newaxis]

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
def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)

    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):

    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())





def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        returns = []
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        numberofbug = 0
        ep = 0
        bug_flags = [False, False, False, False]
        numberbugs = 0
        lebug = []
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                if key == 'info':
                    continue
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.

            for t in range(flags.unroll_length):
                # while not env_output['done'][0][0]:
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                timings.time("model")
                env_output = env.step(agent_output["action"])


                # print('ennnnvvvn',numberofbug,env_output['reward'])
                timings.time("step")
                fi = open('/scratch/nstevia/torchbeastppowuji/10outputrew.txt', 'a+')
                fi.write(str(numberbugs) + ',' + str(env_output['reward']) + ',' + str(ep) + ',' + str(
                    actor_index) + ',' + str(env_output['info']) + os.linesep)
                fi.close()
                if env_output['done'][0][0]:
                    ep=ep+1
                for key in env_output:
                    if key == 'info':
                        continue
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")

            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        rewards_p = []
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        rewards = batch["reward"]
        discounted_reward = 0
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards
        for reward, is_terminal in zip(reversed(rewards.cpu().numpy()), reversed(batch["done"].cpu().numpy())):

            if np.all(is_terminal):
                discounted_reward = 0
            discounted_reward = reward + (flags.gamma * discounted_reward)
            rewards_p.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards_p = torch.tensor(rewards_p, dtype=torch.float32).to(device=flags.device)
        rewards_p = (rewards_p - rewards_p.mean()) / (rewards_p.std() + 1e-7)
        #advantages = rewards_p.detach() - batch['baseline']

        for _ in range(flags.K_epochs):
            learner_outputs, unused_state = model(batch, initial_agent_state)


            # Take final value function slice for bootstrapping.
            #test={key: tensor for key, tensor in learner_outputs.items()}
            learner_outputs = {key: tensor for key, tensor in learner_outputs.items()} #{key: tensor[:-1] for key, tensor in learner_outputs.items()}
            entropy_loss = flags.entropy_cost * compute_entropy_loss(
                learner_outputs["policy_logits"]
            )


            #torch.Size([])
            #torch.Size([79, 32, 9]) learner_outputs["policy_logits"].shape
            #torch.Size([80, 32]) rewards.shape
            #torch.Size([80, 32]) batch["done"].shape
            #torch.Size([80, 32, 9]) batch['policy_logits'].shape
            #torch.Size([80, 32]) advantages.shape
            #torch.Size([80, 32]) rewards_p.shape
            #torch.Size([79, 32]) learner_outputs["baseline"].shape
            advantages = rewards_p.detach() - learner_outputs['baseline']
            ratios = torch.exp(learner_outputs["heur"] - batch['heur'])

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - flags.eps_clip, 1 + flags.eps_clip) * advantages
            #print(torch.min(surr1, surr2).shape,D.Categorical(logits=learner_outputs["heur"]).entropy().shape,learner_outputs["heur"].shape)
            total_loss = -torch.min(surr1, surr2) + nn.MSELoss()(learner_outputs["baseline"],
                                                           rewards_p) - flags.entropy_cost * torch.mean(D.Categorical(logits=learner_outputs["policy_logits"]).entropy())

            #print('leshapes', total_loss, learner_outputs["policy_logits"].shape,learner_outputs["heur"].shape, rewards.shape,
            #      batch["done"].shape,
             #     batch['policy_logits'].shape,batch['heur'].shape, advantages.shape, rewards_p.shape, learner_outputs["baseline"].shape)
            bootstrap_value = learner_outputs["baseline"][-1]
            discounts = (~batch["done"]).float() * flags.discounting



            episode_returns = batch["episode_return"][batch["done"]]
            fi = open('/scratch/nstevia/torchbeastppowuji/10outputepisodes.txt', 'a+')
            fi.write(
                str('begin') + ',' + str(len(episode_returns)) + ',' + str(tuple(episode_returns.cpu().numpy())) + ',' + str() + os.linesep)
            fi.close()
            stats = {
                "episode_returns": tuple(episode_returns.cpu().numpy()),
                "mean_episode_return": torch.mean(episode_returns).item(),
                "total_loss": total_loss.mean().item(),
                "pg_loss": None,
                "baseline_loss": None,
                "entropy_loss": entropy_loss.item(),
            }

            optimizer.zero_grad()
            total_loss.mean().backward()
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
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),

        baseline=dict(size=(T + 1,), dtype=torch.float32),
        heur=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        act_e=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    train_start = time.time()
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    model = Net(env.observation_space.shape, env.action_space.n, flags.use_lstm)
    checkpoint_pretrain = torch.load('/scratch/nstevia/palaas/torchbeast/IMPALA_Pretrained/model.tar')
    for name, target_param in model.named_parameters():
        for param in checkpoint_pretrain["model_state_dict"]:
            if param==name:
                target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                           param].shape == target_param.shape  else print(
                    checkpoint_pretrain["model_state_dict"][param].shape,
                    target_param.shape)
    # checkpoint_pretrain = torch.load('/scratch/nstevia/torchbeast/palaas/torchbeast/IMPALA_Pretrained/model.tar')
    # for name, target_param in model.named_parameters():
    #     for param in checkpoint_pretrain["model_state_dict"]:
    #         if param==name:
    #             target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
    #                                                                                        param].shape == target_param.shape  else print(
    #                 checkpoint_pretrain["model_state_dict"][param].shape,
    #                 target_param.shape)
    #model.load_state_dict(checkpoint["model_state_dict"])
    buffers = create_buffers(flags, env.observation_space.shape, model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

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
    #optimizer.load_state_dict(checkpoint_pretrain["optimizer_state_dict"])
    #scheduler.load_state_dict(checkpoint_pretrain["scheduler_state_dict"])

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B
            if step>=10000:
                break

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
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

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
            if step>=10000:
                break
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    fi = open('/scratch/nstevia/torchbeastppowuji/11traintime.txt', 'a+')
    fi.write(str(time.time() - train_start) +  os.linesep)
    fi.close()

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpointpath = flags.savedir + "/latest/model.tar"
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_start = time.time()
    total_rew = []
    st=0
    observation = env.initial()
    lebug = []
    numberbugs = 0
    agent_state = model.initial_state(batch_size=1)
    while st<300000:


        agent_outputs = model(observation,agent_state)
        policy_outputs, _ = agent_outputs

        observation = env.step(policy_outputs["action"])
        total_rew.append(observation['reward'])
        print(st,observation['info'])
        if observation['info']['bug'] is not None:
            if observation['info']['bug'] not in lebug:
                numberbugs = numberbugs + 1
                lebug.append(observation['info']['bug'])
                fi = open('/scratch/nstevia/torchbeastppowuji/10typeofbugs_addtrain.txt', 'a+')
                fi.write(str(observation['reward']) + ',' + str(observation['info']['bug']) + ',' + str(numberbugs) + os.linesep)
                fi.close()
        #if observation["done"].item():
        if observation["done"].item() or st % 400 == 0:
            done = observation["done"].item()
            observation = env.initial()

            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
        st=st+1
    fi = open('/scratch/nstevia/torchbeastppowuji/10evaltime.txt', 'a+')
    fi.write(str(time.time() - eval_start) + ',' +str(total_rew )+ os.linesep)
    fi.close()
    plogger.close()


def test(flags, num_episodes: int = 10):
    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpointpath = flags.savedir + "/latest/model.tar"
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_start = time.time()
    total_rew = []
    st=0
    observation = env.initial()
    lebug = []
    numberbugs = 0
    agent_state = model.initial_state(batch_size=1)
    while st<3000000:


        agent_outputs = model(observation,agent_state)
        policy_outputs, _ = agent_outputs

        observation = env.step(policy_outputs["action"])
        total_rew.append(observation['reward'])
        print(observation['info']['bug'])
        if observation['info']['bug'] is not None:
            if observation['info']['bug'] not in lebug:
                numberbugs = numberbugs + 1
                lebug.append(observation['info']['bug'])
                fi = open('/scratch/nstevia/torchbeastppowuji/10typeofbugs_addtrain.txt', 'a+')
                fi.write(str(st)+','+str(observation['reward']) + ',' + str(observation['info']['bug']) + ',' + str(numberbugs) + os.linesep)
                fi.close()
        #if observation["done"].item():
        if observation["done"].item() or st % 400 == 0:
            done = observation["done"].item()
            observation = env.initial()
            agent_state = model.initial_state(batch_size=1)

            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
        st=st+1
    fi = open('/scratch/nstevia/torchbeastppowuji/10evaltime_test.txt', 'a+')
    fi.write(str(time.time() - eval_start) + ',' +str(total_rew )+ os.linesep)
    fi.close()

class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False,device=None):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=1,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1
        self.device = device
        self.is_finetune = False
        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)
    def fineModel(self,is_finetune):
        if not is_finetune:
            core_output_size = self.fc.out_features + self.num_actions + 1

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
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)  # x.view(T * B, -1)
        self.fc = nn.Linear(x.shape[1], 512, device=self.device)

        self.fineModel(self.is_finetune)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        #clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        clipped_reward = inputs["reward"].view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

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
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        policy_logits=F.softmax(policy_logits, dim=-1)
        baseline = self.baseline(core_output)

        if self.training:
            print('policy_tra', policy_logits.shape)
            #action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
            act_e= D.Categorical(logits=policy_logits).sample()
            action=act_e
            heur = D.Categorical(logits=policy_logits).log_prob(act_e)
            policy_logits = policy_logits.view(T, B, self.num_actions)
            heur = heur.view(T, B)
        else:
            # Don't sample when testing.
            print('policy_logits',policy_logits)
            action = torch.argmax(policy_logits, dim=1)
            act_e=action
            heur = None
            policy_logits = policy_logits.view(T, B, self.num_actions)


        act_e=act_e.view(T,B)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action,act_e=act_e
                 ,heur=heur),
            core_state,
        )


Net = AtariNet


def create_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            EnvM(), #atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
