import math
import torch
import datetime
import torch.nn as nn
import torch
from skimage.draw import random_shapes
import gym
from gym.spaces import Box, Discrete
import gc
gc.collect()
torch.cuda.empty_cache()
import collections
import functools
import json
import os
import torch.utils.data.distributed

from Environment import LTREnvV2
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
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg, vec_evaluate_episode_rtg_evaluate
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
        self.returnrr=False

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
        obs = self.env.reset()
        self.obs_stack.append(obs)
        self.act_stack.append(0)
        self.rew_stack.append(0)
        self.done_stack.append(0)
        self.info_stack.append(None)
        return self._get_obs()

    def step(self, action):
        """Replaces env observation with fixed length observation history."""
        # Update applied action to the previous timestep.
        self.act_stack[-1] = action
        if self.returnrr:
            obs,  reward, done, info,rr,map = self.env.step(action,return_rr=self.returnrr)
        else:
            obs, reward, done, info= self.env.step(action, return_rr=self.returnrr)
        #if self.jpeg_obs:
        #    obs = self._process_jpeg(obs)
        self.rew_stack[-1] = reward
        # Update frame stack.
        self.obs_stack.append(obs)

        #self.act_stack.append(0)  # Append unknown action to current timestep.
        #self.rew_stack.append(0)
        self.info_stack.append(None)
        if self.returnrr:
            return self._get_obs(), reward, done, None,rr,map
        return self._get_obs(), reward, done, None

    def pad_current_episode(self, obs, n):
        # Prepad current episode with n steps.
        for _ in range(n):
            self.obs_stack.append(np.zeros_like(obs))
            self.act_stack.append(0)
            self.rew_stack.append(0)
            self.done_stack.append(1)
            self.info_stack.append(None)



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




import torch

# --- Create offline RL dataset
MAX_EPISODE_LEN = 1000
# --- Create model
from multigame_dt import MultiGameDecisionTransformer
MAX_EPISODE_LEN = 1000
# --- Evaluate model



from replay_buffer import ReplayBuffer


class Experiment:
    def __init__(self, variant,parser,env,path_prefix,final_directory,eval_loc):
        self.options = parser.parse_args()
        ngpus_per_node = torch.cuda.device_count()
        self.path_prefix=path_prefix

        """ This next line is the key to getting DistributedDataParallel working on SLURM:
        		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
         		current process inside a node and is also 0 or 1 in this example."""
        OBSERVATION_SHAPE = (31, 1025) #(84, 84) (20, 20)
        PATCH_SHAPE = (5, 5) # The size of tensor a (16) must match the size of tensor b (36) at non-singleton dimension 2
        NUM_ACTIONS = 31  # 18 Maximum number of actions in the full dataset.
        # rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
        NUM_REWARDS = 5
        RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.target_entropy = -self.act_dim
        #self.logger = Logger(variant)
        self.eval_loc=eval_loc
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
        self.target_net = MultiGameDecisionTransformer(
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
        self.parser = parser
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        from load_pretrained import load_jax_weights
        if self.eval_loc == 1:
            self.results_test = os.path.join(final_directory, 'results_testn.txt')
            self.box_test = os.path.join(final_directory, 'box_testn.txt')
            self.finalr = os.path.join(final_directory, 'finalrn.txt')
            fi = open(self.finalr, 'a+')
            fi.write('mrr,actualrank' + '\n')
            fi.close()
            fi = open(self.results_test, 'a+')
            fi.write('algo,project_name,reward,time,episode,steps,mrr,map,actualr,top1,top5,top10' + '\n')
            fi.close()
            fi = open(self.box_test, 'a+')
            fi.write('algo,project_name,bug_id,mrr,map,top1,top5,top10' + '\n')
            fi.close()
        else:
            results_train = os.path.join(final_directory, 'results_train.txt')

            fi = open(results_train, 'a+')
            fi.write('algo,project_name,reward,time,episode,steps' + '\n')
            fi.close()
            self.results_train = results_train
        if self.eval_loc==1:
            if self.options.finetune=='0_f':
                model_params, model_state = pickle.load(open(
                    "/scratch/f/foutsekh/nstevia/Harnessing_generalist_agents_on_SE/MGDT_experiments/bug_localization/models/checkpoint_38274228.pkl",
                    "rb"))
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")  # variant.get("device", "cuda")
                load_jax_weights(self.model, model_params)
            else:
                self._load_model(self.path_prefix)
        else:
            model_params, model_state = pickle.load(open("/scratch/f/foutsekh/nstevia/Harnessing_generalist_agents_on_SE/MGDT_experiments/bug_localization/models/checkpoint_38274228.pkl", "rb"))
            #variant.get("device", "cuda")
            load_jax_weights(self.model, model_params)
            load_jax_weights(self.target_net, model_params)
        self.model = self.model.to(self.device)
        self.state_mean, self.state_std=None,None
        self.replay_buffer = ReplayBuffer(variant["replay_size"])
        self.aug_trajs = []
        self.r=None

        self.numberOfbugs=0
        self.target_entropy = -self.act_dim
        self.buglist=[]
        self.env = SequenceEnvironmentWrapper(env, num_stack_frames=1, jpeg_obs=True)

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001

        self.tempera=self.model.temperature()
        self.entro=self.model.target_entropy
        import time
        self.start_time=time.time()


    def _get_env_spec(self, variant):

        act_dim = 31
        return (31,1025), act_dim, [0,30]

    def _save_model(self, path_prefix, is_pretrain_model=False):
        #torch.save(self.model.state_dict(), f"{path_prefix}/model_as_mgdt.pth")
        is_pretrain_model = False
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
        #path_prefix = '/scratch/nstevia/online-dt-wuji/multigame-dt/exp/2023.07.24/093003-default'
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint_pretrain = torch.load(f)
            for name, target_param in self.model.named_parameters():
                for param in checkpoint_pretrain["model_state_dict"]:
                    if param == name:
                        target_param.data.copy_(checkpoint_pretrain["model_state_dict"][param].data) if \
                        checkpoint_pretrain["model_state_dict"][
                            param].shape == target_param.shape else print(
                            checkpoint_pretrain["model_state_dict"][param].shape,
                            target_param.shape)
            #self.optimizer.load_state_dict(checkpoint_pretrain["optimizer_state_dict"])
            #self.scheduler.load_state_dict(checkpoint_pretrain["scheduler_state_dict"])
            #self.log_temperature_optimizer.load_state_dict(
             #   checkpoint_pretrain["log_temperature_optimizer_state_dict"]
            #)
            #self.pretrain_iter = checkpoint_pretrain["pretrain_iter"]
            # self.online_iter = checkpoint["online_iter"]
            #self.total_transitions_sampled = checkpoint_pretrain["total_transitions_sampled"]
            np.random.set_state(checkpoint_pretrain["np"])
            random.setstate(checkpoint_pretrain["python"])
            torch.set_rng_state(checkpoint_pretrain["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")
        else:
            assert False


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

            trajs,r,t = vec_evaluate_episode_rtg(  #returns, lengths, trajs = vec_evaluate_episode_rtg(
                self.env,
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
        to_use={}
        to_use['rew_all']=r
        to_use['ep_len'] = t
        return to_use
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
        trajs,r,t = vec_evaluate_episode_rtg_evaluate(  #returns, lengths, trajs = vec_evaluate_episode_rtg(
                self.env,
                self.options,
                self.state_dim,
                self.results_test,
                self.model,
                self.box_test,
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
            target=self.target_net,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        maxon=(7500*int(self.variant['finetune'][0]))/100
        import time
        while self.online_iter < maxon:#self.variant["max_online_iters"]:

            start = time.time()
            outputs = {}

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
            self.online_iter = self.online_iter + 1
            print('number of bugs is and online iter is', len(self.replay_buffer.trajectories), self.online_iter)

            train_outputs = trainer.train_iteration(
                    loss_fn=loss_fn,
                    dataloader=dataloader,
                )
            import time
            fi = open(self.results_train, 'a+')
            fi.write(
                self.options.algo + "," + self.options.project_name + "," + str(x['rew_all']) + "," + str(-start + time.time()) + "," + str(
                    self.online_iter) + "," + str(x['ep_len']) + '\n')
            fi.close()
            print(self.options.algo + "," + self.options.project_name + "," + str(x['rew_all']) + "," + str(-start + time.time()) + "," + str(
                    self.online_iter) + "," + str(x['ep_len']))
            outputs.update(train_outputs)
            outputs["time/total"] = time.time() - self.start_time

            self._save_model(
                     path_prefix=self.path_prefix,
                     is_pretrain_model=True,
                 )


    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution

            #torch.Size([8, 1]) torch.Size([8, 1, 31]) torch.Size([8])
            #torch.Size([8, 1]) torch.Size([8, 1, 31]) torch.Size([8, 8])
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )
        eval_envs = None
        if self.eval_loc==1:
            self.evaluate()
        else:

            self.online_tuning(None, eval_envs, loss_fn)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    current_directory = os.getcwd()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default='MsPacman') #hopper-medium-v2

    # model options
    parser.add_argument('--file_path', default=r"C:\Users\phili\Downloads\10428077\Replication\LTR\\", help='File Path') #r"C:\Users\phili\Downloads\10428077\Replication\LTR\\
    parser.add_argument('--cache_path', default=r"C:\Users\phili\Downloads\10428077\Replication\.buffer_cache_ac", help='Cache Path')
    parser.add_argument('--prev_policy_model_path', default=None, help='Trained Policy Path')
    parser.add_argument('--prev_value_model_path', default=None, help='Trained Value Path')
    parser.add_argument('--train_data_path',default=r'/Volumes/Engineering/Harnessing_generalist_agents_on_SE/MGDT_experiments/bug_localization/aspectJ/MAENT/AspectJ_train_for_mining_before.csv', help='Training Data Path')
    parser.add_argument('--save_path',default=r'C:\Users\phili\Downloads\10428077\Replication\LTR\Models\AC\Entropy\AspectJ\\', help='Save Path')
    parser.add_argument('--project_name',default='AspectJ',  help='Project Name')
    parser.add_argument('--algo', default='AC_WITHOUT_ENTROPY', type=str, help='Project Name')
    parser.add_argument('--run', default='1',type=str, help='Project Name')
    parser.add_argument('--finetune', default='1_f', type=str, help='Project Name')
    parser.add_argument('--non_original', default=0, type=int, help='Project Name')
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
    parser.add_argument("--num_updates_per_online_iter", type=int, default=2) #300
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    #parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--eval', default=0, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--estimate', default=False, type=bool, help='Project Name')
    options = parser.parse_args()
    file_path = os.path.join(current_directory,options.finetune, options.run+'_LTR')
    cache_path = os.path.join(file_path, '.buffer_cache_ac')
    prev_policy_model_path = options.prev_policy_model_path
    prev_value_model_path = options.prev_value_model_path
    train_data_path = options.train_data_path# r'C:\Users\phili\Downloads\10428077\Replication\LTR\Models\AC\Entropy\AspectJ\\'
    #train_data_path='/scratch/f/foutsekh/nstevia/bug_localization/RL_Model/bug_loc_experiments/AspectJ_train_for_mining_before.csv'
    project_name = options.project_name
    save_path = os.path.join(file_path, 'Models',options.algo,options.project_name)

    #current_directory = os.getcwd()
    final_directory = os.path.join(current_directory,options.finetune,  options.algo+'_'+options.project_name+'_'+options.run)
    os.makedirs(final_directory, exist_ok=True)


    Path(file_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True,exist_ok=True)
    #used to be data_path=file_path + train_data_path,
    mpath='/scratch/f/foutsekh/nstevia/bug_localization/micro_codebert'
    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    metric=None
    non_original = False
    reg_path=None
    #experiment = Experiment(vars(args))
    at='we are at '+str(options.eval)+' ,'+str(options.finetune)+' ,'+str(options.run)
    print(at)
    if options.eval==1:
        env = LTREnvV2(data_path=train_data_path, model_path=mpath,  # data_path=file_path + test_data_path
                       tokenizer_path=mpath, action_space_dim=31, report_count=None, max_len=512,
                       use_gpu=True, caching=True, file_path=file_path, project_list=[project_name], test_env=True,
                       estimate=options.estimate, metric=metric, non_original=non_original, reg_path=reg_path)
    else:
        env = LTREnvV2(data_path=train_data_path, model_path=mpath,
                       tokenizer_path=mpath, action_space_dim=31, report_count=100, max_len=512,
                       use_gpu=True, caching=True, file_path=file_path, project_list=[project_name],metric=metric,non_original=non_original,reg_path=reg_path)

    experiment = Experiment(vars(args), parser,env,file_path,final_directory,options.eval)
    print("=" * 50)
    experiment()

