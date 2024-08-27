"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import torch
import numpy as np
import random


MAX_EPISODE_LEN = 1000


class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood. We clamp them to be within
        # the user defined action range.
        self.action_range = action_range

    def __call__(self, traj):
        inputs={}
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        #print(traj["observations"].shape,self.state_dim,traj["picked"].shape,traj["actions2"].shape,traj["returns-to-go"].shape,traj["rewards2"].shape,"ooooo")
        ss = traj["observations"][si : si + self.max_len,:,:].reshape((-1,)+self.state_dim) #traj["observations"][si : si + self.max_len].reshape(-1, self.state_dim)
        inputs["observations"]=traj["observations"][si : si + self.max_len,:,:]

        aa = traj["actions"][si : si + self.max_len].reshape(-1, 1)#traj["actions"][si : si + self.max_len].reshape(-1, self.act_dim)
        inputs["actions"]=traj["actions2"][si : si + self.max_len]
        inputs["picked"] = traj["picked"][si: si + self.max_len, :]
        inputs["returns-to-go"]=traj["returns-to-go"][si : si + self.max_len,:]
        rr = traj["rewards"][si : si + self.max_len].reshape(-1, 1)
        inputs["rewards"] = traj["rewards2"][si : si + self.max_len]
        if "terminals" in traj:
            dd = traj["terminals"][si : si + self.max_len]  # .reshape(-1)
        else:
            dd = traj["dones"][si : si + self.max_len]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = inputs["observations"].shape[0] #ss.shape[0]

        timesteps = np.arange(si, si + tlen)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff

        rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        act_len = inputs["actions"].shape[0] #aa.shape[0]
        if tlen != act_len:
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen,)+ self.state_dim), ss]) #np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), ss])
        #ss = (ss - self.state_mean) / self.state_std
        #print('icicicicic',self.act_dim,aa.shape,(self.max_len,tlen, self.act_dim))  18 (20, 1) (20, 20, 18)
        aa = np.concatenate([np.zeros((self.max_len - tlen, 1)), aa])#np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa])
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr])
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 2, dd])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg])
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps])
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(tlen), np.ones(inputs["actions"].shape[0])]) #np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])
        ss = torch.from_numpy(ss).to(dtype=torch.float32)
        aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*self.action_range)
        rr = torch.from_numpy(rr).to(dtype=torch.float32)
        dd = torch.from_numpy(dd).to(dtype=torch.long)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long)
        ordering = torch.from_numpy(ordering).to(dtype=torch.long)
        padding_mask = torch.from_numpy(padding_mask)

        return ss, aa, rr, dd, rtg, timesteps, ordering, padding_mask,inputs


def collate_fn(batch):
  #print(len(batch))
  inp={'actions':[],'observations':[],'rewards':[],'returns-to-go':[],'picked':[]}
  inps={'actions':[],'observations':[],'rewards':[],'returns-to-go':[]}
  don=[]
  actions=[]
  observations=[]
  rewards=[]
  returns_to_go=[]
  actions_n=[]
  observations_n=[]
  rewards_n=[]
  returns_to_go_n=[]
  for x in batch:
      #print(x)
      (ss, aa, rr, dones, rtg, timesteps, ordering, padding_mask, inputs) = x

      #for a in dones:
       #   don.append(torch.from_numpy(np.array(a)))
      for a in inputs:
          for aa in range(inputs[a].shape[0]):
              #print(a,inputs[a].shape)
              if len(inputs[a].shape)==1:
                  inp[a].append(torch.from_numpy(np.array([inputs[a][aa]])))
              else:
                inp[a].append(torch.from_numpy(inputs[a][aa]))
      #for a in inputs_n:
      #    for aa in range(inputs_n[a].shape[0]):
       #       inps[a].append(torch.from_numpy(inputs_n[a][aa]))

  #don=torch.stack(don)

  inputs={a:torch.stack(inp[a]) for a in inp}
  #inputs_n = {a: torch.stack(inps[a]) for a in inps}



  return None, None, None, None,None , None, None, None,inputs
def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    num_workers=24, #24
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size)

    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
    )

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size,collate_fn=collate_fn, num_workers=1, shuffle=True
    )
    #return torch.utils.data.DataLoader(
    #    subset, batch_size=1, num_workers=2, shuffle=False
    #)torch.utils.data.DataLoader(
     #   subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    #)


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size):

    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)

    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds
