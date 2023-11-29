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
import cv2
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
            .reshape(num_envs, state_dim[0],state_dim[1])
            .to(device=device, dtype=torch.float32)
        ).reshape(num_envs, -1, state_dim[0],state_dim[1])
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
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim[0],state_dim[1])
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

def check_bug1():
    folder_bug = 'bug_left/'
    files = os.listdir(folder_bug)
    img_bug = [file for file in files if file.startswith('bug')]
    img = Image.open("current_screen.png")
    left = 0
    top = 90
    right = 15
    bottom = 120
    im1 = img.crop((left, top, right, bottom))
    im1.save('current_test.png')
    imgA = cv2.imread("current_test.png")
    for elem in img_bug:
        imgB = cv2.imread(folder_bug + elem)
        s = ssim(imgA, imgB, multichannel=True)
        if s > 0.9:
            print(s)
            return True
    return False



def check_bug3():
    folder_bug = 'bug_left/'
    files = os.listdir(folder_bug)
    img_bug = [file for file in files if file.startswith('bug')]
    img = Image.open("current_screen.png")
    left = 0
    top = 42
    right = 15
    bottom = 72
    im1 = img.crop((left, top, right, bottom))
    im1.save('current_test.png')
    imgA = cv2.imread("current_test.png")
    for elem in img_bug:
        imgB = cv2.imread(folder_bug + elem)
        s = ssim(imgA, imgB, multichannel=True)
        if s > 0.9:
            print(s)
            return True
    return False



def check_bug2():
    folder_bug = 'bug_right/'
    files = os.listdir(folder_bug)
    img_bug = [file for file in files if file.startswith('bug')]
    img = Image.open("current_screen.png")
    left = 305
    top = 90
    right = 320
    bottom = 120
    im1 = img.crop((left, top, right, bottom))
    im1.save('current_test.png')
    imgA = cv2.imread("current_test.png")
    for elem in img_bug:
        imgB = cv2.imread(folder_bug + elem)
        s = ssim(imgA, imgB, multichannel=True)
        if s > 0.9:
            print(s)
            return True
    return False



def check_bug4():
    folder_bug = 'bug_right/'
    files = os.listdir(folder_bug)
    img_bug = [file for file in files if file.startswith('bug')]
    img = Image.open("current_screen.png")
    left = 305
    top = 42
    right = 320
    bottom = 72
    im1 = img.crop((left, top, right, bottom))
    im1.save('current_test.png')
    imgA = cv2.imread("current_test.png")
    for elem in img_bug:
        imgB = cv2.imread(folder_bug + elem)
        s = ssim(imgA, imgB, multichannel=True)
        if s > 0.9:
            print(s)
            return True
    return False
def vec_evaluate_episode_rtg(
create_env,
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
    mode="eval",
    use_mean=False,
    image_in=True,
):
    #assert len(target_return) == vec_env.num_envs

    
    #model.to(device=device)
    numberbugs=0
    #state_mean = torch.from_numpy(state_mean).to(device=device)
    #state_std = torch.from_numpy(state_std).to(device=device)

    #num_envs = vec_env.num_envs
    #state = vec_env.reset()
    RETURN_RANGE = [-20, 100]

    env_name = "MsPacman"
    num_envs = 1
    env_fn = lambda: create_env(env_name)

    envs = [env_fn() for _ in range(num_envs)]
    num_episodes=1
    trajectories = []

    num_batch = len(envs)
    num_steps = envs[0].spec.max_episode_steps
    if mode == 'normal':
        envs[0].evaluate = False
    else:
        envs[0].evaluate = True
    if not envs[0].evaluate:
        num_steps=27000
    #num_steps=1000
    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_episodes)]

    rew_sum_list = []
    for c in range(1): #num_episodes // num_batch
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
        total_rew=[]
        t=0
        while True: #for t in range(num_steps)

            done_prev = done
            obs = {k: torch.tensor(v, device=device) for k, v in obs.items()}
            o.append(obs['observations'].cpu().numpy()[0])
            a.append(obs['actions'].cpu().numpy()[0])
            r.append(obs['rewards'].cpu().numpy()[0])
            d.append(done_prev[0])
            # torch.Size([1, 4, 1, 84, 84]) torch.Size([1, 4]) torch.Size([1, 4]) torch.Size([1, 4])
            actions, _, _, rg = model.optimal_action(obs,  return_range=RETURN_RANGE,
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
            total_rew.append(obs['rewards'][0])
            traj['rewards'] = np.append(traj['rewards'], rew)
            t=t+1
            if np.all(done) or t>num_steps:
                
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
                if mode=='eval':
                    f = open('5bug_log_RELINE_addtrain11k.txt', 'a+')
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

    #return (
     #   episode_return.reshape(num_envs),
     #   episode_length.reshape(num_envs),
     #   trajectories,
   # )
