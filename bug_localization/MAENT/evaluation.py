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
# from skimage.metrics import structural_similarity as ssim

import collections
from mb_agg import *
from agent_utils import *

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


def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)


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

    # model.eval()

    numberbugs = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    env_name = ""
    num_envs = 1
    env_fn = create_env

    envs = [env_fn]
    num_episodes = 1

    trajectories = []

    num_batch = len(envs)
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_episodes)]

    rew_sum_list = []
    if mode == 'normal':
        for c in range(1):  # num_episodes // num_batch
            o, a, r, d, rtg, adj, fea, mask, cand, topick = [], [], [], [], [], [], [], [], [], []
            seeds = seeds_list[c * num_batch: (c + 1) * num_batch]
            rng.manual_seed(seeds[0])
            bug_flags = [False, False, False, False]
            obs_list = [env.reset() for i, env in enumerate(envs)]

            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

            done = np.zeros(num_batch, dtype=np.int32)
            traj = {'observations': np.array([]), 'actions2': np.array([]), 'rewards': np.array([]),
                    'terminals': np.array([]), 'rewards2': np.array([]), 'actions': np.array([]),
                    "returns-to-go": np.array([]), 'adj': np.array([]), 'fea': np.array([]), 'mask': np.array([]),
                    "picked": np.array([]), }
            traj['actions'] = np.append(traj['actions'], 0)
            traj['rewards'] = np.append(traj['rewards'], 0)
            total_rew = 0
            t = 0
            lebug = []
            obs = envs[0].reset()

            pik = []

            xx = torch.from_numpy(to_one_hot(pik, max_size=envs[0].env.action_space.n)).to(device).type(torch.float)
            topick.append(xx.cpu().numpy()[0])
            while True:  # for t in range(num_steps)

                done_prev = done
                obs = {k: torch.tensor(v).to(device) for k, v in
                       obs.items()}  # {k: torch.tensor(v).cuda() for k, v in obs.items()}
                o.append(obs['observations'].cpu().numpy()[0])
                a.append(obs['actions'].cpu().numpy()[0])
                r.append(obs['rewards'].cpu().numpy()[0])

                d.append(done_prev[0])

                # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])
                temp_action = torch.from_numpy(to_one_hot(pik, max_size=envs[0].env.action_space.n)).to(device).type(
                    torch.float)
                topick.append(temp_action.cpu().numpy()[0])
                actions, _, _, rg = model.optimal_action(obs, temp_action, return_range=RETURN_RANGE,
                                                         single_return_token=True,
                                                         opt_weight=0,
                                                         num_samples=128,
                                                         action_temperature=1.0,
                                                         return_temperature=0.75,
                                                         action_top_percentile=50,
                                                         return_top_percentile=None, rng=rng, deterministic=False)
                rtg.append(rg.cpu().numpy())
                obs, rew, done, _ = envs[0].step(actions.item())
                total_rew = total_rew + rew
                pik.append(actions.item())
                traj['actions'] = np.append(traj['actions'], actions.cpu().numpy())
                #print(done, rew, 'evaluation177', actions.item())
                done = np.logical_or(done, done_prev).astype(np.int32)
                rew = rew * (1 - done)

                traj['rewards'] = np.append(traj['rewards'], rew)
                t = t + 1

                if np.all(done):
                    o.append(obs['observations'][0])
                    a.append(obs['actions'][0])
                    r.append(obs['rewards'][0])
                    rtg.append(torch.zeros_like(rg).cpu().numpy())
                    d.append(1)

                    traj['actions2'] = np.stack((p for p in a), axis=0)
                    traj['observations'] = np.stack((p for p in o), axis=0)
                    traj['rewards2'] = np.stack((p for p in r), axis=0)
                    traj["returns-to-go"] = np.stack((p for p in rtg), axis=0)
                    traj["picked"] = np.stack((p for p in topick), axis=0)
                    traj['terminals'] = np.array(d).astype(bool)
                    trajectories.append(traj)

                    break

    return trajectories, total_rew, t

    # return (
    #   episode_return.reshape(num_envs),
    #   episode_length.reshape(num_envs),
    #   trajectories,
# )
def vec_evaluate_episode_rtg_evaluate(
        create_env,
        options,
        state_dim,
        results_test,
        model,
        box_test,
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

    # model.eval()

    numberbugs = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    env_name = ""
    num_envs = 1
    env_fn = create_env

    envs = [env_fn]
    num_episodes = envs[0].env.suppoerted_len

    trajectories = []

    num_batch = len(envs)
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_episodes)]

    rew_sum_list = []
    if mode == 'eval':
        import time
        start = time.time()
        st = 0
        total_e = 0
        all_map = []
        map_per_r = []
        final_top = [0, 0, 0]
        all_rr = []
        for c in range(envs[0].env.suppoerted_len):  # num_episodes // num_batch
            total_e += 1
            rew_all = 0
            irr = False
            envs[0].returnrr=True
            all_rr.append(-100)
            o, a, r, d, rtg, adj, fea, mask, cand, topick = [], [], [], [], [], [], [], [], [], []
            seeds = seeds_list[c * num_batch: (c + 1) * num_batch]
            rng.manual_seed(seeds[0])
            bug_flags = [False, False, False, False]
            obs_list = [env.reset() for i, env in enumerate(envs)]

            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}

            done = np.zeros(num_batch, dtype=np.int32)
            traj = {'observations': np.array([]), 'actions2': np.array([]), 'rewards': np.array([]),
                    'terminals': np.array([]), 'rewards2': np.array([]), 'actions': np.array([]),
                    "returns-to-go": np.array([]), 'adj': np.array([]), 'fea': np.array([]), 'mask': np.array([]),
                    "picked": np.array([]), }
            traj['actions'] = np.append(traj['actions'], 0)
            traj['rewards'] = np.append(traj['rewards'], 0)
            t_done=False
            #total_rew = 0
            t = 0
            lebug = []
            obs = envs[0].reset()

            pik = []

            xx = torch.from_numpy(to_one_hot(pik, max_size=envs[0].env.action_space.n)).to(device).type(torch.float)
            topick.append(xx.cpu().numpy()[0])
            while not t_done:  # for t in range(num_steps)

                done_prev = done
                obs = {k: torch.tensor(v).to(device) for k, v in
                       obs.items()}  # {k: torch.tensor(v).cuda() for k, v in obs.items()}
                o.append(obs['observations'].cpu().numpy()[0])
                a.append(obs['actions'].cpu().numpy()[0])
                r.append(obs['rewards'].cpu().numpy()[0])

                d.append(done_prev[0])

                # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])
                temp_action = torch.from_numpy(to_one_hot(pik, max_size=envs[0].env.action_space.n)).to(device).type(
                    torch.float)
                topick.append(temp_action.cpu().numpy()[0])
                actions, _, _, rg = model.optimal_action(obs, temp_action, return_range=RETURN_RANGE,
                                                         single_return_token=True,
                                                         opt_weight=0,
                                                         num_samples=128,
                                                         action_temperature=1.0,
                                                         return_temperature=0.75,
                                                         action_top_percentile=50,
                                                         return_top_percentile=None, rng=rng, deterministic=False)
                rtg.append(rg.cpu().numpy())
                obs, rew, done, _,rr,map = envs[0].step(actions.item())#rr,map = env.step(action, return_rr=True)
                rew_all = rew_all + rew
                if all_rr[-1] < rr:
                    all_rr[-1] = rr
                pik.append(actions.item())
                traj['actions'] = np.append(traj['actions'], actions.cpu().numpy())
                #print(done, rew, 'evaluation177', actions.item())
                done = np.logical_or(done, done_prev).astype(np.int32)
                rew = rew * (1 - done)

                traj['rewards'] = np.append(traj['rewards'], rew)
                t = t + 1

                if np.all(done):
                    o.append(obs['observations'][0])
                    a.append(obs['actions'][0])
                    r.append(obs['rewards'][0])
                    rtg.append(torch.zeros_like(rg).cpu().numpy())
                    d.append(1)

                    traj['actions2'] = np.stack((p for p in a), axis=0)
                    traj['observations'] = np.stack((p for p in o), axis=0)
                    traj['rewards2'] = np.stack((p for p in r), axis=0)
                    traj["returns-to-go"] = np.stack((p for p in rtg), axis=0)
                    traj["picked"] = np.stack((p for p in topick), axis=0)
                    traj['terminals'] = np.array(d).astype(bool)
                    #trajectories.append(traj)
                    t_done=True
            real_fix = 0
            precision_at_k = []
            print('next')
            print(envs[0].env.picked)
            print(envs[0].env.match_id)
            print('nextnnnn')
            for kk in range(len(envs[0].env.picked)):
                if envs[0].env.picked[kk] in envs[0].env.match_id:
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

                        if len(envs[0].env.picked) >= uu + 1:
                            if envs[0].env.picked[uu] in envs[0].env.match_id:
                                cc += 1
                    top.append(cc)

            # algo,project_name,bug_id,mrr,map,top1,top5,top10' box_test,results_test
            ret = all_rr[-1] if all_rr[-1] > 0 else -1
            fi = open(box_test, 'a+')
            fi.write(
                options.algo + "," + options.project_name + "," + str(total_e) + "," + str(ret) + "," + str(
                    map_per_r[-1]) + "," + str(top[0]) +
                "," + str(top[1]) + "," + str(top[2]) + '\n')
            fi.close()
            final_top[0], final_top[1], final_top[2] = final_top[0] + top[0], final_top[1] + top[1], final_top[2] + top[
                2]
            fi = open(results_test, 'a+')
            fi.write(
                options.algo + "," + options.project_name + "," + str(rew_all) + "," + str(-start + time.time()) + "," + str(
                    total_e) + "," + str(st) + "," + str(temp.mean()) + "," + str(
                    np.mean(np.array(map_per_r))) + "," + str(final_top[0] / envs[0].env.suppoerted_len) + "," + str(
                    final_top[1] / envs[0].env.suppoerted_len) + "," + str(final_top[2] / envs[0].env.suppoerted_len) + '\n')
            fi.close()


    return None, rew_all, None

    # return (
    #   episode_return.reshape(num_envs),
    #   episode_length.reshape(num_envs),
    #   trajectories,