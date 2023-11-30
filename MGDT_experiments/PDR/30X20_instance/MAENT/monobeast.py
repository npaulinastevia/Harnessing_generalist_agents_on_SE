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
from torchbeast.validation import validate
from torchbeast.mb_agg import *
from torchbeast.agent_utils import *
from gym.spaces import Box, Discrete
from skimage.draw import random_shapes
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from PIL import Image, ImageFile
from Params import configs
#ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchbeast.JSSP_Env import SJSSP
from skimage.metrics import structural_similarity as ssim
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import cv2
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from Params import configs
from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace
import numpy as np
from torchbeast.models.graphcnn_congForSJSSP import GraphCNN
#from torchbeast.wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
 #   VonNeumannMotion

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
parser.add_argument("--savedir", default="/scratch/nstevia/torchbeastimpalal2d/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=48, type=int, metavar="N",
                    help="Number of actors (default: 4).")###
parser.add_argument("--total_steps", default=1000000, type=int, metavar="T",
                    help="Total environment steps to train for.")###
parser.add_argument("--batch_size", default=32, type=int, metavar="B",
                    help="Learner batch size.")#3#
parser.add_argument("--core", default=6400, type=int, metavar="B",
                    help=".")
parser.add_argument("--unroll_length", default=40, type=int, metavar="T",#80
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", default=True,action="store_true",
                    help="Use LSTM in agent model.")

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
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]
device = torch.device(configs.device)
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




def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""

    policy = F.softmax(logits, dim=1)#policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=1)#log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):

    cross_entropy = F.nll_loss(
        logits,#F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
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
        from torchbeast.uniform_instance_gen import uni_instance_gen
        data_generator = uni_instance_gen
        dataLoaded = np.load('/scratch/nstevia/torchbeastimpalal2d/torchbeast/DataGen/generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(
            configs.np_seed_validation) + '.npy')
        vali_data = []
        for i in range(dataLoaded.shape[0]):
            vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))
        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        #env_output['adj']=env_output['adj'].to_sparse()
        agent_state = model.initial_state(batch_size=1)
        g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                 batch_size=torch.Size([1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                 n_nodes=configs.n_j * configs.n_m,
                                 device='cpu')
        padded_nei = None

        agent_output, unused_state = model(env_output['fea'], g_pool_step, padded_nei,env_output['adj'],env_output['candidate'].unsqueeze(0),
                                           env_output['mask'].unsqueeze(0),agent_state,env_output)
    
        ep = 0
        agent_state=unused_state
        numberbugs = 0
        lebug = []
        #initial_agent_state_buffers=[]
        flags.core=model.core.num_layers
        #for _ in range(flags.num_buffers):

         #   state = tuple(torch.zeros(model.core.num_layers, 1, model.core.hidden_size) for _ in range(2)
        #)

         #   for t in state:
          #      t.share_memory_()

           # initial_agent_state_buffers.append(state)

            #i=tuple(
           # torch.zeros(model.core.num_layers, 1, model.core.hidden_size)
            #for _ in range(2)
        #)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            print(index,'index')
            for key in env_output:
                if key == 'info' or key == 'frame':
                    continue

                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:

                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i]=tensor

                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.

            ep=ep+1
            for t in range(flags.unroll_length):
            #while not env_output['done'][0][0]:
                timings.reset()

                with torch.no_grad():

                    agent_output, agent_state = model(env_output['fea'], g_pool_step, padded_nei,env_output['adj'],env_output['candidate'].unsqueeze(0),
                                           env_output['mask'].unsqueeze(0),agent_state,env_output)
                timings.time("model")
                env_output = env.step(agent_output["action"])

                #env_output['adj'] = env_output['adj'].to_sparse()
                #print('ennnnvvvn',numberofbug,env_output['reward'])
                timings.time("step")
                fi = open('/scratch/nstevia/torchbeastimpalal2d/18outputrewOPT.txt', 'a+')
                fi.write(str(numberbugs) + ',' + str(env_output['reward'])+ ',' + str(ep) + ',' + str(actor_index)+',' +str(env_output['info'])+  os.linesep)
                fi.close()
                for key in env_output:
                    if key=='info' or key == 'frame':
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

        
        
        padded_nei = None
        #model(env_output['fea'], g_pool_step, padded_nei, env_output['adj'], env_output['candidate'].unsqueeze(0),
         #     env_output['mask'].unsqueeze(0), agent_state, env_output)
        batch['fea']=batch['fea'].view(-1,batch['fea'].shape[-1])
        batch['adj']=batch['adj'].view(-1,batch['adj'].shape[-2],batch['adj'].shape[-1])
        #batch['adj'] =batch['adj'].to(device='cpu')
        mb_g_pool = g_pool_cal(configs.graph_pool_type, batch['adj'].to(device).shape, configs.n_j * configs.n_m,
                               torch.device(configs.device))
        temp = aggr_obs(torch.stack(tuple(x for x in batch['adj'])).to(device), configs.n_j*configs.n_m).to(device=batch['adj'].device)

        model.batch_size=batch['adj'].shape[0]
        model.initial = True
        batch['adj']=temp
        batch['candidate']=batch['candidate'].view(-1,batch['candidate'].shape[-1])
        batch['mask']=batch['mask'].view(-1,batch['mask'].shape[-1])

        learner_outputs, unused_state = model(batch['fea'],
                                        mb_g_pool,
                                              padded_nei,
                                        batch['adj'],
                                        batch['candidate'],
                                        batch['mask'],
                                              initial_agent_state,
                                        batch)#model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.

        bootstrap_value = learner_outputs["baseline"][-1]
        batch['adj']=batch['adj'].to_dense()
        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        #print('line244',batch.keys())
        batch = {key: tensor for key, tensor in batch.items()}#{key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor for key, tensor in learner_outputs.items()} #{key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action_P"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action_P"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["pol_log"]#learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        fi = open('/scratch/nstevia/torchbeastimpalal2d/18outputepisodesOPT.txt', 'a+')
        fi.write(
            str('begin') + ',' + str(len(episode_returns)) + ',' + str(tuple(episode_returns.cpu().numpy())) + ',' + str() + os.linesep)
        fi.close()
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
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    from torchbeast.uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen
    env=SJSSP(n_j=configs.n_j, n_m=configs.n_m)
    num_actions=1
    adj, fea, candidate, mask = env.reset(
        data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))

    specs = dict(
        frame=dict(size=(T + 1,), dtype=torch.uint8),
        adj=dict(size=(T + 1,*adj.shape), dtype=torch.float32),
        fea=dict(size=(T + 1, *fea.shape), dtype=torch.float32),
        candidate=dict(size=(T + 1,*candidate.shape), dtype=torch.int64),
        mask=dict(size=(T + 1,*mask.shape), dtype=torch.bool),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, *candidate.shape), dtype=torch.float32),
        pol_log=dict(size=(T + 1, *candidate.shape), dtype=torch.float32),
        baseline=dict(size=(T + 1,*candidate.shape), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
        action_P=dict(size=(T + 1,), dtype=torch.int64),
        

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

    model = Net(env.observation_space, 1, flags.use_lstm)#Net(env.observation_space.shape, env.action_space.n, flags.use_lstm)
    checkpoint_pretrain = torch.load('/scratch/nstevia/torchbeastimpalal2d/palaas/torchbeast/torchbeast-20230515-132901/model.tar')
    for name, target_param in model.named_parameters():
        for param in checkpoint_pretrain["model_state_dict"]:
            if param==name:
                target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if checkpoint_pretrain["model_state_dict"][
                                                                                           param].shape == target_param.shape  else print(
                    checkpoint_pretrain["model_state_dict"][param].shape,
                    target_param.shape)

    #model.load_state_dict(checkpoint["model_state_dict"])
    buffers = create_buffers(flags, env.observation_space, model)#create_buffers(flags, env.observation_space.shape, model.num_actions)

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

    learner_model = Net(env.observation_space, 1, flags.use_lstm).to(device=torch.device(configs.device))
    learner_model.device=torch.device(configs.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    # initial_agent_state_buffers = []
    # for _ in range(flags.num_buffers):
    #     state = tuple(torch.zeros(flags.core, 1, flags.core) for _ in range(2)
    #     )
    #     initial_agent_state_buffers.append(state)
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
    fi = open('/scratch/nstevia/torchbeastimpalal2d/18traintimeOPT.txt', 'a+')
    fi.write(str(time.time() - train_start) +  os.linesep)
    fi.close()
    N_JOBS_P = configs.n_j
    # params.Pn_j
    N_MACHINES_P = configs.n_m  # params.Pn_m

    N_JOBS_N = configs.n_j  # params.Nn_j
    N_MACHINES_N = configs.n_m  # params.Nn_m
    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device='cpu')
    padded_nei = None


    model = Net(gym_env.observation_space, 1, flags.use_lstm)
    model.eval()
    checkpointpath = flags.savedir + "/latest/model.tar"
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_start = time.time()
    total_rew = []
    st=0


    lebug = []
    numberbugs = 0
    from torchbeast.uniform_instance_gen import uni_instance_gen
    data_generator = uni_instance_gen

    dataLoaded = np.load(
        '/scratch/nstevia/torchbeastimpalal2d/torchbeast/DataGen/generatedData' + str(configs.n_j) + '_' + str(
            configs.n_m) + '_Seed' + str(
            configs.np_seed_validation) + '.npy')
    dataset = []
    for i in range(dataLoaded.shape[0]):
        dataset.append((dataLoaded[i][0], dataLoaded[i][1]))
    for i, data in enumerate(dataset):
        agent_state = model.initial_state(batch_size=1)
        #adj, fea, candidate, mask = env.reset(data)
        observation = env.initial(data=data)
        ep_reward = - env.gym_env.max_endTime
        while True:
            #agent_outputs = model(observation)
            agent_output, agent_state = model(observation['fea'], g_pool_step, padded_nei, observation['adj'],
                                               observation['candidate'].unsqueeze(0),
                                               observation['mask'].unsqueeze(0), agent_state, observation)
            policy_outputs = agent_output
            observation = env.step(policy_outputs["action"])
            ep_reward += observation['reward']
            total_rew.append(ep_reward)
            print(observation["done"].item(),env.gym_env.posRewards,-ep_reward + env.gym_env.posRewards,'st')
            fi = open('/scratch/nstevia/torchbeastimpalal2d/7testresults'+str(configs.n_j)+str(configs.n_m)+'.txt', 'a+')
            fi.write(str(ep_reward) +','+str(-ep_reward + env.gym_env.posRewards)+ os.linesep)
            fi.close()
            if observation["done"].item():
                done = observation["done"].item()
                #observation = env.initial()
                logging.info(
                    "Episode ended after %d steps. Return: %.1f",
                    observation["episode_step"].item(),
                    observation["episode_return"].item(),
                )
                break

    fi = open('/scratch/nstevia/torchbeastimpalal2d/18evaltimeOPT.txt', 'a+')
    fi.write(str(time.time() - eval_start) + ',' +str(total_rew )+ os.linesep)
    fi.close()
    plogger.close()


def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_env(flags)
    env = environment.Environment(gym_env)
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n, flags.use_lstm)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []

    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )


class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions,use_lstm=False, n_j=configs.n_j,
              n_m=configs.n_m,
                 learn_eps=False,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,  device=None):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        # Feature extraction.
        #self.conv1 = nn.Conv2d(
        #    in_channels=self.observation_shape[0],
         #   out_channels=32,
          #  kernel_size=8,
          #  stride=1,
        #)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.batch_size=None
        self.n_ops_perjob = n_m
        self.device=device
        self.is_finetune=False
        self.use_lstm = use_lstm
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1
        self.initial=True


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
        self.batch_size=batch_size
        if not self.use_lstm:
            return tuple()
        return list(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, x,graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,core_state=(),inputs={}):
        #print(x.dtype,adj.dtype,candidate.dtype,mask.dtype,'500')#torch.Size([1000, 2]) torch.Size([1000, 1000]) torch.Size([1, 50]) torch.float32 torch.int32 torch.int32 torch.bool

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)

        # prepare policy feature: concat omega feature with global feature

        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1)).to(device=self.device)

        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy).to(device=self.device)

        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature).to(device=self.device)

        '''# prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_j, self.n_m))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(omega, torch.ones_like(omega, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob'''
        # concatenate feature
        # concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        x = torch.cat((candidate_feature, h_pooled_repeated), dim=-1).to(device=self.device)

        #x = inputs["frame"]  # [T, B, C, H, W].
        T=1
        B, B2, *_ = x.shape

        #x = torch.flatten(x, 0, 1)  # Merge time and batch.
        #x = x.float() / 255.0
        #x = F.relu(self.conv1(x))nn.Linear(20, 30)
        #input = torch.randn(128, 20) 128, 30
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #x = nn.Flatten()(x)#x.view(T * B, -1)
        self.fc = nn.Linear(x.shape[-1], 32,device=self.device)
        self.fineModel(self.is_finetune)
         #torch.Size([1, 50, 128]) Linear(in_features=50, out_features=512, bias=True
        x = F.relu(self.fc(x))
        #one_hot_last_action = F.one_hot(
        #    inputs["last_action"].view(T * B), self.num_actions
        #).float()
        clipped_reward = inputs["reward"].view(T * B, 1)
        #print(x.shape,clipped_reward.shape, one_hot_last_action.shape,self.fc)
        t1,t2,*_=x.shape
        x=x.view(x.shape[0],-1)
        #core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)
        core_input = x#torch.cat([x], dim=-1)

        if self.use_lstm:

            core_output_list = []
            core_input = core_input.view(T, B, -1)
            self.core = nn.LSTM(core_input.shape[-1], core_input.shape[-1], 2, device=self.device)
            self.core2 = nn.LSTM(h_pooled.shape[-1], h_pooled.shape[-1], 2, device=self.device)
            #self.batch_size=core_state.shape[-2]

            if self.initial:
                core_state=tuple(
                    torch.zeros(self.core.num_layers, self.batch_size, self.core.hidden_size).to(device=self.device)
                    for _ in range(2)
                )
                self.initial=False
            notdone = (~inputs["done"]).float()
            notdone=notdone.view(-1)

            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:

                nd = nd.view(-1, 1, 1).to(device=self.device)#nd.view(1, -1, 1)

                core_state = tuple(nd * s for s in core_state)

                output, core_state = self.core(input.unsqueeze(0), core_state)

                core_output_list.append(output)


            core_output = torch.flatten(torch.cat(core_output_list), 0, 1).to(device=self.device)
        else:
            core_output = core_input
            core_state = tuple()

        core_output=core_output.view(B,B2,-1)

        self.policy = nn.Linear(core_output.shape[-1], self.num_actions, device=self.device)

        policy_logits = self.policy(core_output)
        mask_reshape = mask.reshape(policy_logits.size()).to(device=self.device)
        policy_logits[mask_reshape] = -1e5#float('-inf')
        #print(core_output.shape,'coreoutput')
        pi = F.softmax(policy_logits, dim=1)
        #pi=policy_logits
        core_output=core_output.view(core_output.shape[0],-1)
        self.baseline = nn.Linear(core_output.shape[-1], 1, device=self.device)
        baseline = self.baseline(core_output)

        if self.training:
            if pi.shape[0]==1:
                action,idx=select_action(pi, inputs['candidate'].to(device=self.device), None)
                action_P=idx
            else:
                action=action_P=None
            #action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action_P=None
            action = greedy_select_action(pi,inputs['candidate'])
            #action_P=torch.argmax(pi, dim=1)
        #policy_logits = policy_logits.view(T, B, self.num_actions)
        #baseline = baseline.view(T, B)
        #action = action.view(T, B)

        return (
            dict(policy_logits=torch.squeeze(pi), baseline=torch.squeeze(baseline), action=action,action_P=action_P
                 , pol_log=torch.squeeze(policy_logits)),
            list(core_state),
        )


Net = AtariNet


def create_env(flags):
    from torchbeast.JSSP_Env import SJSSP
    return SJSSP(n_j=configs.n_j, n_m=configs.n_m)
    #atari_wrappers.wrap_pytorch(
     #   atari_wrappers.wrap_deepmind(
     #       SJSSP(n_j=configs.n_j, n_m=configs.n_m) , #atari_wrappers.make_atari(flags.env), EnvM()
      #      clip_rewards=False,
       #     frame_stack=True,
       #     scale=False,
       # )
    #)


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
