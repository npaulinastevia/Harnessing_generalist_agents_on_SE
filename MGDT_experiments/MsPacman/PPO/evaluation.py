"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import functools
import math
import os
import random

import numpy
from PIL import Image
import numpy as np
import torch
# from skimage.metrics import structural_similarity as ssim

import collections

# import cv2
MAX_EPISODE_LEN = 1000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

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

    if mode == 'eval':
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

    model.train()
    #model  # model.to(device=device)
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
    num_steps = envs[0].spec.max_episode_steps
    #envs[0].evaluate = True
    if mode=='normal':
        envs[0].evaluate = False
    else:
        envs[0].evaluate=True
    if not envs[0].evaluate:
        num_steps=27000
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_episodes)]

    rew_sum_list = []
    for c in range(1):  # num_episodes // num_batch
        o, a,aval, r, d, rtg, oold, aold, rold, dold,st,ac = [], [], [], [], [], [], [], [], [],[],[],[]
        seeds = seeds_list[c * num_batch: (c + 1) * num_batch]
        rng.manual_seed(seeds[0])

        obs_list = [env.reset() for i, env in
                    enumerate(envs)]  # [env.reset(seed=seeds[i]) for i, env in enumerate(envs)]
        obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

        done = np.zeros(num_batch, dtype=np.int32)
        traj = {'observations': np.array([]), 'actions2': np.array([]), 'rewards': np.array([]),
                'terminals': np.array([]), 'rewards2': np.array([]), 'actions': np.array([]), 'stateval': np.array([]),
                "returns-to-go": np.array([]), 'observationsold': np.array([]), 'actions2old': np.array([]),
                'rewardsold': np.array([]),
                'terminalsold': np.array([]), 'rewards2old': np.array([]), 'actionsold': np.array([]),
                "returns-to-goold": np.array([]),"actlogprob": np.array([])}
        total_rew = []
        st.append(np.zeros((4,1)))
        ac.append(np.zeros(4))
        t = 0
        while True:  # for t in range(num_steps)

            done_prev = done
            o.append(obs['observations'][0])
            a.append(obs['actions'][0])
            r.append(obs['rewards'][0])
            d.append(done_prev[0])
            traj['actionsold'] = np.append(traj['actionsold'], obs['actions'][0][-2])
            traj['rewardsold'] = np.append(traj['rewardsold'], np.array([obs['rewards'][0][-2]]))
            obs = {k: torch.tensor(v).cuda() for k, v in
                   obs.items()}  # {k: torch.tensor(v, device=device) for k, v in obs.items()}
            # d.append(done_prev[0])
            # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])

            actions, _, logprob, rg, act_val = model.optimal_action(obs, return_range=RETURN_RANGE,
                                                                         single_return_token=True,
                                                                         opt_weight=0,
                                                                         num_samples=128,
                                                                         action_temperature=1.0,
                                                                         return_temperature=0.75,
                                                                         action_top_percentile=50,
                                                                         return_top_percentile=None, rng=rng,
                                                                         deterministic=False)
            rtg.append(rg.cpu().numpy()[0])
            # Collect step results and stack as a batch.
            step_results = [env.step(act) for env, act in zip(envs, actions.cpu().numpy())]
            obs_list = [result[0] for result in step_results]
            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
            # o.append(obs['observations'][0])
            # a.append(obs['actions'][0])
            # r.append(obs['rewards'][0])
            rew = np.stack([result[1] for result in step_results])
            done = np.stack([result[2] for result in step_results])
            traj['actions'] = np.append(traj['actions'], actions.cpu().numpy()[0])
            st.append(act_val.detach().cpu().numpy()[0])
            ac.append(logprob.detach().cpu().numpy()[0])
            done = np.logical_or(done, done_prev).astype(np.int32)
            # d.append(done[0])
            rew = rew * (1 - done)
            traj['rewards'] = np.append(traj['rewards'], rew)
            total_rew.append(obs['rewards'][0])
            t = t + 1
            if np.all(done) or t==num_steps-1:
                o.append(obs['observations'][0])
                a.append(obs['actions'][0])
                r.append(obs['rewards'][0])
                rtg.append(torch.zeros_like(rg).cpu().numpy()[0])
                d.append(1)
                traj['actions2'] = np.stack((p for p in a), axis=0)
                traj['observations'] = np.stack((p for p in o), axis=0)
                traj['stateval'] = np.stack((p for p in st), axis=0)
                traj['actlogprob'] = np.stack((p for p in ac), axis=0)
                traj['rewards2'] = np.stack((p for p in r), axis=0)
                traj["returns-to-go"] = np.stack((p for p in rtg), axis=0)
                traj['terminals'] = np.array(d).astype(bool)
                # traj['actions2old'] = np.stack((p for p in aold), axis=0)
                # traj['observationsold'] = np.stack((p for p in oold), axis=0)
                # traj['rewards2old'] = np.stack((p for p in rold), axis=0)
                traj["returns-to-go"] = np.stack((p for p in rtg), axis=0)
                trajectories.append(traj)
                print(len(lebug),'len lebug')
                f = open('3bug_log_RELINE10k.txt', 'a+')
                if envs[0].bug_flags[0]:
                    f.write('BUG1 ')
                if envs[0].bug_flags[1]:
                    f.write('BUG2 ')
                if envs[0].bug_flags[2]:
                    f.write('BUG3 ')
                if envs[0].bug_flags[3]:
                    f.write('BUG4 ')
                f.write('\n')
                f.close()


                break

    return trajectories, envs[0].numberbugs,total_rew,t

    # return (
    #   episode_return.reshape(num_envs),
    #   episode_length.reshape(num_envs),
    #   trajectories,
# )
