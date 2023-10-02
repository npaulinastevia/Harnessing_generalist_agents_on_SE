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
"""The environment class for MonoBeast."""

import torch
from Params import configs
import numpy as np
def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        
        self.episode_return = None
        self.episode_step = None
        from torchbeast.uniform_instance_gen import uni_instance_gen
        self.data_generator = uni_instance_gen

        dataLoaded = np.load('/scratch/nstevia/torchbeastppol2d/torchbeast/DataGen/generatedData' + str(configs.n_j) + '_' + str(configs.n_m) + '_Seed' + str(
            configs.np_seed_validation) + '.npy')
        vali_data = []
        for i in range(dataLoaded.shape[0]):
            vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

    def initial(self,data=None):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)

        adj, fea, candidate, mask = self.gym_env.reset(self.data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))
        if data is not None:
            adj, fea, candidate, mask = self.gym_env.reset(data)
        initial_frame=None
        return dict(
            frame=initial_frame,
            adj=torch.from_numpy(np.copy(adj)),
            fea=torch.from_numpy(fea),
            candidate=torch.from_numpy(candidate),
            mask=torch.from_numpy(mask),
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        adj, fea, reward, done, candidate, mask = self.gym_env.step(action.item())
        unused_info=None
        frame=None
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            adj, fea, candidate, mask = self.gym_env.reset(
                self.data_generator(n_j=configs.n_j, n_m=configs.n_m, low=configs.low, high=configs.high))
            #frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        #frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            frame=frame,
            adj=torch.from_numpy(np.copy(adj)),
            fea=torch.from_numpy(fea),
            candidate=torch.from_numpy(candidate),
            mask=torch.from_numpy(mask),
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            info=unused_info,
        )

    def close(self):
        self.gym_env.close()
