import uuid

import numpy as np
import gym
import pandas as pd
from gym import spaces
import torch
import zlib
import random
#from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import math
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.predict=None
    def forward(self, x):
        out=self.linear(x)
        self.predict=out
        out = torch.sigmoid(out)
        return out
class LTREnv(gym.Env):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, max_len=512, use_gpu=True,
                 file_path="", project_list=None, test_env=False,estimate=False,non_original=True,reg_path=None):
        super(LTREnv, self).__init__()
        #use_gpu=True
        if use_gpu:
            self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.dev = "cpu"
        self.current_file = None
        self.file_path = file_path
        self.test_env = test_env
        self.current_id = None
        self.df = None  # done
        self.sampled_id = None  # done
        self.filtered_df = None  # done
        self.max_len = max_len
        self.project_list = project_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(tokenizer_path).to(self.dev)
        self.data_path = data_path
        self.action_space_dim = action_space_dim
        self.action_space = spaces.Discrete(self.action_space_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(1025,), dtype=np.float32)

        self.report_count = report_count
        self.non_original=non_original
        self.previous_obs = None
        self.picked = []
        self.remained = []
        self.estimate=estimate
        self.irr=None
        self.t = 0
        modelr = LogisticRegression(2)#'/scratch/f/foutsekh/nstevia/regression_model_develop/Aspectj/
        if self.non_original:
            print(self.non_original,'self.non_original')
            state_dict = torch.load(
                reg_path+'final_logistic_regression_model.pt')  # (file_path + model_path)
            modelr.load_state_dict(state_dict=state_dict)
            self.regrem = modelr.to(self.dev)
        #self.scaler=StandardScaler()
        self.suppoerted_len = None
        self.match_id = None
        self.__get_ids()
        self.counter = 0

    @staticmethod
    def decode(text):
        return zlib.decompress(bytes.fromhex(text)).decode()

    @staticmethod
    def reduce_dimension_by_mean_pooling(embeddings, attention_mask, to_numpy=False):
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings.detach() * mask
        summed = torch.sum(masked_embeddings, 2)
        summed_mask = torch.clamp(mask.sum(2), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled.numpy() if to_numpy else mean_pooled

    def __get_ids(self):

        self.df = pd.read_csv(self.data_path)

        #dff=pd.read_excel("Tomcat.xlsx")#Tomcat.xlsx SWT.xlsx

        # Unnamed: 10




        #irr=self.df[self.df['cid'] == self.picked[-1]]['file_content'].tolist()



        #self.df = self.df.assign(project_name=['AspectJ']*len(self.df['bug_id'].tolist()))
        #self.df = self.df.assign(report=self.df['description'].tolist())
        #self.df = self.df.assign(cid=[uuid.uuid1().int]*len(self.df['file_content'].tolist()))

        if self.test_env:
            self.df = self.df.drop(
                columns=["summary", "description", "report_time", "report_timestamp", "status", "commit",
                         "commit_timestamp",
                          ], axis=1)
            #columns = ["summary", "description", "report_time", "report_timestamp", "status", "commit",
            #           "commit_timestamp", "files", "Unnamed: 10", "bug_recency", "report_id", "rVSM_similarity",
             #          "bug_frequency", "classname_similarity", "file", "collab_filter"]
            matched = self.df[self.df['match'] == 1]

            if self.project_list[0]=='JDT':
                not_matched = pd.DataFrame(columns=matched.columns)
                match_counter = matched.groupby('id')['cid'].count()
                for row in match_counter.iteritems():
                    temp = self.df[(self.df['match'] != 1) & (self.df['id'] == row[0])].sample(frac=1).reset_index(
                        drop=True).head(self.action_space_dim - row[1])
                    not_matched = pd.concat([not_matched, temp])
                self.df = pd.concat([matched, not_matched]).sample(frac=1).reset_index(drop=True)
            else:
                not_matched = pd.DataFrame(columns=matched.columns)
                match_counter = matched.groupby('id')['cid'].count()
                for row in match_counter.iteritems():
                    temp = self.df[(self.df['match'] != 1) & (self.df['id'] == row[0])].sample(frac=1).reset_index(
                        drop=True).head(self.action_space_dim - row[1])
                    not_matched = pd.concat([not_matched, temp])
                self.df = pd.concat([matched, not_matched]).sample(frac=1).reset_index(drop=True)

        if self.project_list is not None:
            self.df = self.df[self.df['project_name'].isin(self.project_list)].reset_index(drop=True)


        #if not self.estimate:
         #   self.df = self.df[self.df['report'].notna()]

        if self.estimate:
            self.irr = list(set(self.df[self.df['file_content'].isnull()]['id'].tolist()))
        if not self.test_env:
            matched = self.df[self.df['match'] == 1]

            not_matched = pd.DataFrame(columns=matched.columns)
            match_counter = matched.groupby('id')['cid'].count()
      
            for row in match_counter.iteritems():#match_counter.iteritems():
                temp = self.df[(self.df['match'] != 1) & (self.df['id'] == row[0])].sample(frac=1).reset_index(
                    drop=True).head(self.action_space_dim - row[1])

                not_matched = pd.concat([not_matched, temp])
            self.df = pd.concat([matched, not_matched]).sample(frac=1).reset_index(drop=True)
        id_list = self.df.groupby('id')['cid'].count()
        id_list = id_list[id_list == int(self.action_space_dim)].index.to_list()


        self.suppoerted_len = len(id_list)

        
        if self.report_count is None:
            self.report_count = self.suppoerted_len
        id_list = self.df[(self.df['id'].isin(id_list)) & (self.df['match'] == 1)]['id'].unique().tolist()
        #random.seed(59)  # 13
        #print(id_list,'dddddododod',self.report_count,self.suppoerted_len)

        self.sampled_id = random.sample(id_list, min(len(id_list), self.report_count))

        

    def reset(self):
        self.previous_obs = None
        #self.current_id = random.sample(self.sampled_id, 1)[0] if self.counter >= len(self.sampled_id) else self.sampled_id[self.counter]
        self.current_id = self.sampled_id[self.counter % len(self.sampled_id)]

        self.counter += 1
        self.__get_filtered_df()
        self.picked = []
        self.remained = self.filtered_df['cid'].tolist()
        self.match_id = self.filtered_df[self.filtered_df['match'] == 1]['cid'].tolist()
        self.t = 0
        # self.picked.append(random.sample(self.remained, 1)[0])
        # self.remained.remove(self.picked[-1])
        return self.__get_observation()
    # def __calculate_reward(self, return_rr=False):
    #     if self.t == 0:
    #         return 0, 0 if return_rr else 0, None
    #     else:
    #         current_matches = [self.df[self.df['cid'] == item]['match'].tolist()[0] for item in self.picked]
    #         current_average_precision = average_precision(current_matches)
    #         reward = current_average_precision
    #         if return_rr:
    #             positions = self.df[self.df['cid'].isin(self.picked)]['match'].to_numpy()
    #             max_position = np.argmax(positions) + 1 if any(positions == 1) else -1
    #             return reward, 1.0 / max_position
    #         return reward, None
    def __calculate_reward(self, return_rr=False):

        if self.t == 0:
            return 0, 0,0 if return_rr else 0, None,None
        else:
            relevance = self.df[self.df['cid'] == self.picked[-1]]['match'].tolist()[0]
            already_picked = any(self.df[self.df['cid'].isin(self.picked)]['match'].tolist())

            # ------------------************************------------------------
            current_matches = [self.df[self.df['cid'] == item]['match'].tolist()[0] for item in self.picked]
            current_average_precision = average_precision(current_matches)
            indices_of_match = np.argwhere(np.array(current_matches) == 1)
            distances = (np.insert(indices_of_match, 0, 0) - np.insert(indices_of_match, len(indices_of_match), 0))[
                        1:-1]
            distances = 1 if (len(distances) == 0 or np.any(np.isnan(distances))) else distances.mean()
            # ---------------------********************************----------------------------
            if already_picked:
                reward = (3.0 * relevance) / (np.log2(self.t + 1) * distances) if relevance == 1 else 0
            else:
                reward = -np.log2(self.t + 1)
            if return_rr:
                positions = self.df[self.df['cid'].isin(self.picked)]['match'].to_numpy()
                max_position = np.argmax(positions) + 1 if any(positions == 1) else -1

                return reward, 1.0 / max_position,current_average_precision

            return reward, None,None

    def step(self, action, return_rr=False):
        temp = self.filtered_df['cid'].tolist()[action]
        info = {"invalid": False}
        obs, reward, done = None, None, None
        if self.non_original:
            inp = torch.tensor([self.filtered_df['churn'].tolist()[action], self.filtered_df['pre_r_bugs'].tolist()[action]], dtype=torch.float32).to(self.dev)
            mediary=self.regrem(inp)
            metric=mediary.item()#self.regrem.predict.item()#self.filtered_df[self.metric].tolist()[action] 'churn', 'pre_r_bugs'
        else:
            metric=0
        if temp in self.remained:
            self.picked.append(temp)
            self.remained.remove(temp)
            obs = self.__get_observation()
            reward, rr,map = self.__calculate_reward(return_rr=return_rr)
            done = self.t == len(self.filtered_df)

        else:
            info['invalid'] = True
            obs = self.previous_obs
            rr = -1
            done = True  # self.t == len(self.filtered_df)
            # ToDo: Check it
            reward = -6 if len(self.picked) < self.action_space.n else 0
            reward = reward + metric
            _, _, map = self.__calculate_reward(return_rr=return_rr)


        if return_rr:

            return obs, reward, done, info, rr,map
        return obs, reward, done, info

    def __get_filtered_df(self):

        self.filtered_df = self.df[self.df["id"] == self.current_id].reset_index()

        self.filtered_df = self.filtered_df.sample(frac=1, random_state=int(self.filtered_df['cid'].sum())).reset_index(
            drop=True)

    def __get_observation(self):
        # ToDO: This will not be worked

        self.t += 1
        report_data, code_data = self.df[
                                     (self.df['cid'] == self.picked[-1]) & (self.df['id'] == self.current_id)].report, \
                                 self.df[
                                     (self.df['cid'] == self.picked[-1]) & (
                                             self.df['id'] == self.current_id)].file_content
        report_data, code_data = report_data.values.tolist()[0], code_data.values.tolist()[0]
        report_token, code_token = self.tokenizer.batch_encode_plus([report_data], max_length=self.max_len,
                                                                    pad_to_multiple_of=self.max_len,
                                                                    truncation=True,
                                                                    padding=True,
                                                                    return_tensors='pt'), \
                                   self.tokenizer.batch_encode_plus([self.decode(code_data)], max_length=self.max_len,
                                                                    pad_to_multiple_of=self.max_len,
                                                                    truncation=True,
                                                                    padding=True,
                                                                    return_tensors='pt')
        report_output, code_output = self.model(**report_token.to(self.dev)), self.model(**code_token.to(self.dev))
        report_embedding, code_embedding = self.reduce_dimension_by_mean_pooling(report_output.last_hidden_state,
                                                                                 report_token['attention_mask']), \
                                           self.reduce_dimension_by_mean_pooling(code_output.last_hidden_state,
                                                                                 code_token['attention_mask'])
        final_rep = np.concatenate([report_embedding, code_embedding, [[self.t]]], axis=1)[0]
        self.previous_obs = final_rep

        assert False
        return final_rep

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("Current ranking of the documents")
        for i, item in enumerate(self.picked):
            print("Ranking: {} Document Cid: {} Timestep: {}".format(i + 1, item, self.t))


class LTREnvV2(LTREnv):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, max_len=512, use_gpu=True,
                 caching=False, file_path="", project_list=None, test_env=False,estimate=False,metric=None,non_original=False,reg_path=None):
        super(LTREnvV2, self).__init__(data_path, model_path, tokenizer_path, action_space_dim, report_count,
                                       max_len=max_len, use_gpu=use_gpu, file_path=file_path, project_list=project_list,
                                       test_env=test_env,estimate=estimate,non_original=non_original,reg_path=reg_path)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(31,1025), dtype=np.float32)
        self.all_embedding = []
        self.caching = caching
        self.metric=metric



    def reset(self):
        self.all_embedding = []
        return super(LTREnvV2, self).reset()

    def _LTREnv__get_observation(self):
        self.t += 1

        if len(self.all_embedding) == 0:
            if not self.caching or not Path(
                    self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id)).is_file():
                for row in self.filtered_df.iterrows():
                    report_data, code_data = row[1].report, row[1].file_content
                    report_token, code_token = self.tokenizer.batch_encode_plus([report_data], max_length=self.max_len,
                                                                                pad_to_multiple_of=self.max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt'), \
                                               self.tokenizer.batch_encode_plus([self.decode(code_data)],
                                                                                max_length=self.max_len,
                                                                                pad_to_multiple_of=self.max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt')
                    report_output, code_output = self.model(**report_token.to(self.dev)), self.model(
                        **code_token.to(self.dev))
                    report_embedding, code_embedding = self.reduce_dimension_by_mean_pooling(
                        report_output.last_hidden_state,
                        report_token['attention_mask']), \
                                                       self.reduce_dimension_by_mean_pooling(
                                                           code_output.last_hidden_state,
                                                           code_token['attention_mask'])
                    final_rep = np.concatenate([report_embedding.cpu().numpy(), code_embedding.cpu().numpy(), [[1e-7]]], axis=1)[0]
                    self.all_embedding.append(final_rep)
                if self.caching:
                    Path(self.file_path + ".caching/").mkdir(parents=True, exist_ok=True)
                    np.save(self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id),
                            self.all_embedding)

            else:
                self.all_embedding = np.load(
                    self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id)).tolist()
        if len(self.picked) > 0:
            action_index = self.filtered_df['cid'].tolist().index(self.picked[-1])
            self.all_embedding[action_index] = np.full_like(self.all_embedding[action_index], 0,
                                                            dtype=np.double)  # np.zeros_like(self.all_embedding[action_index])
            stacked_rep = np.stack(self.all_embedding)
            stacked_rep[action_index, -1] = self.t
        else:
            stacked_rep = np.stack(self.all_embedding)
        self.previous_obs = stacked_rep
        return stacked_rep


class LTREnvV3(LTREnv):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, max_len=512, use_gpu=True,
                 caching=False, file_path="", project_list=None, test_env=False):
        super(LTREnvV3, self).__init__(data_path, model_path, tokenizer_path, action_space_dim, report_count,
                                       max_len=max_len, use_gpu=use_gpu, file_path=file_path, project_list=project_list,
                                       test_env=test_env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(31, 1025), dtype=np.float32)
        self.all_embedding = []
        self.caching = caching

    def reset(self):
        self.all_embedding = []
        return super(LTREnvV3, self).reset()

    def _LTREnv__get_observation(self):
        self.t += 1

        if len(self.all_embedding) == 0:
            if not self.caching or not Path(
                    self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id)).is_file():
                for row in self.filtered_df.iterrows():
                    report_data, code_data = row[1].report, row[1].file_content
                    report_token, code_token = self.tokenizer.batch_encode_plus([report_data], max_length=self.max_len,
                                                                                pad_to_multiple_of=self.max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt'), \
                                               self.tokenizer.batch_encode_plus([self.decode(code_data)],
                                                                                max_length=self.max_len,
                                                                                pad_to_multiple_of=self.max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt')
                    with torch.no_grad():
                        report_output, code_output = self.model(**report_token.to(self.dev)), self.model(
                            **code_token.to(self.dev))

                    final_rep = np.concatenate([report_output.last_hidden_state, code_output.last_hidden_state], axis=2)
                    self.all_embedding.append(final_rep)
                self.all_embedding = np.stack(self.all_embedding)
                if self.caching:
                    Path(self.file_path + ".caching/").mkdir(parents=True, exist_ok=True)
                    np.save(self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id),
                            self.all_embedding)
            else:
                self.all_embedding = np.load(
                    self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id))
        if len(self.picked) > 0:
            try:
                action_index = self.filtered_df['cid'].tolist().index(self.picked[-1])
                self.all_embedding[action_index, :, :, 768:] = np.full_like(
                    self.all_embedding[action_index, :, :, 768:], 0,
                    dtype=np.double)  # np.zeros_like(self.all_embedding[action_index])
            except Exception as ex:
                print(ex, action_index, self.current_id)
                raise ex
        self.previous_obs = self.all_embedding
        # print("Current_Id", self.current_id, type(self.all_embedding))
        # print(self.all_embedding.shape)
        return self.all_embedding.squeeze(1)


class LTREnvV4(LTREnv):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, report_max_len=512,
                 code_max_len=4096, use_gpu=True,
                 caching=False, file_path="", project_list=None, test_env=False, window_size=480):
        super(LTREnvV4, self).__init__(data_path, model_path, tokenizer_path, action_space_dim, report_count,
                                       max_len=report_max_len, use_gpu=use_gpu, file_path=file_path,
                                       project_list=project_list,
                                       test_env=test_env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(31, (code_max_len % window_size) + code_max_len + report_max_len, 768), dtype=np.float32)
        self.window_size = window_size
        self.embedding_size = 512
        self.report_max_len = report_max_len
        self.code_max_len = code_max_len
        self.all_embedding = []
        self.caching = caching

    def reset(self):
        self.all_embedding = []
        return super(LTREnvV4, self).reset()

    def get_window(self, **kwargs):
        for window_start in range(0, math.ceil(self.code_max_len / self.window_size) * self.window_size,
                                  self.window_size):
            temp = dict()
            for key, value in kwargs.items():
                temp[key] = value[:, window_start: window_start + self.embedding_size].to(self.dev)
            # print(window_start, window_start + self.embedding_size)
            yield temp

    def _LTREnv__get_observation(self):
        self.t += 1

        if len(self.all_embedding) == 0:
            if not self.caching or not Path(
                    self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id)).is_file():
                for row in self.filtered_df.iterrows():
                    report_data, code_data = row[1].report, row[1].file_content
                    report_token, code_token = self.tokenizer.batch_encode_plus([report_data],
                                                                                max_length=self.report_max_len,
                                                                                pad_to_multiple_of=self.report_max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt'), \
                                               self.tokenizer.batch_encode_plus([self.decode(code_data)],
                                                                                max_length=self.code_max_len,
                                                                                pad_to_multiple_of=self.code_max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt')
                    with torch.no_grad():
                        output_part = []
                        for part in self.get_window(**code_token):
                            output_part.append(self.model(**part).last_hidden_state)
                        code_embedding = torch.hstack(output_part)
                        report_embedding = self.model(**report_token.to(self.dev)).last_hidden_state
                    final_rep = np.concatenate([report_embedding, code_embedding], axis=1)
                    self.all_embedding.append(final_rep)
                self.all_embedding = np.stack(self.all_embedding)
                if self.caching:
                    Path(self.file_path + ".caching/").mkdir(parents=True, exist_ok=True)
                    np.save(self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id),
                            self.all_embedding)
            else:
                self.all_embedding = np.load(
                    self.file_path + ".caching/{}_all_embedding.npy".format(self.current_id))
        if len(self.picked) > 0:
            try:
                action_index = self.filtered_df['cid'].tolist().index(self.picked[-1])
                self.all_embedding[action_index, :, self.report_max_len:, :] = np.full_like(
                    self.all_embedding[action_index, :, self.report_max_len:, :], 0,
                    dtype=np.double)  # np.zeros_like(self.all_embedding[action_index])
            except Exception as ex:
                print(ex, action_index, self.current_id)
                raise ex
        self.previous_obs = self.all_embedding
        # print("Current_Id", self.current_id, type(self.all_embedding))
        # print(self.all_embedding.shape)
        return self.all_embedding.squeeze(1)
class LTREnvV5(LTREnv):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, report_max_len=512,
                 code_max_len=4096, use_gpu=True,
                 caching=False, file_path="", project_list=None, test_env=False, window_size=480):
        super(LTREnvV5, self).__init__(data_path, model_path, tokenizer_path, action_space_dim, report_count,
                                       max_len=report_max_len, use_gpu=use_gpu, file_path=file_path,
                                       project_list=project_list,
                                       test_env=test_env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(31, 768), dtype=np.float32)
        self.window_size = window_size
        self.embedding_size = 512
        self.report_max_len = report_max_len
        self.code_max_len = code_max_len
        self.all_embedding = []
        self.caching = caching

    def reset(self):
        self.all_embedding = []
        return super(LTREnvV5, self).reset()

    def get_window(self, **kwargs):
        for window_start in range(0, math.ceil(self.code_max_len / self.window_size) * self.window_size,
                                  self.window_size):
            temp = dict()
            for key, value in kwargs.items():
                temp[key] = value[:, window_start: window_start + self.embedding_size].to(self.dev)
            # print(window_start, window_start + self.embedding_size)
            yield temp

    def _LTREnv__get_observation(self):
        self.t += 1

        if len(self.all_embedding) == 0:
            if not self.caching or not Path(
                    self.file_path + ".caching/{}/{}_all_embedding.npy".format(self.project_list[0], self.current_id)).is_file():
                for row in self.filtered_df.iterrows():
                    report_data, code_data = row[1].report, row[1].file_content
                    report_token, code_token = self.tokenizer.batch_encode_plus([report_data],
                                                                                max_length=self.report_max_len,
                                                                                pad_to_multiple_of=self.report_max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt'), \
                                               self.tokenizer.batch_encode_plus([self.decode(code_data)],
                                                                                max_length=self.code_max_len,
                                                                                pad_to_multiple_of=self.code_max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt')
                    with torch.no_grad():
                        output_part = []
                        for part in self.get_window(**code_token):
                            output_part.append(self.model(**part).pooler_output)
                        code_embedding = torch.vstack(output_part)
                        report_embedding = self.model(**report_token.to(self.dev)).pooler_output
                    final_rep = np.concatenate([report_embedding, torch.unsqueeze(code_embedding.mean(dim=0),0)], axis=1)
                    self.all_embedding.append(final_rep)
                self.all_embedding = np.stack(self.all_embedding) # 31,1, 1536
                if self.caching:
                    Path(self.file_path + ".caching/{}/".format(self.project_list[0])).mkdir(parents=True, exist_ok=True)
                    np.save(self.file_path + ".caching/{}/{}_all_embedding.npy".format(self.project_list[0], self.current_id),
                            self.all_embedding)
            else:
                self.all_embedding = np.load(
                    self.file_path + ".caching/{}/{}_all_embedding.npy".format(self.project_list[0], self.current_id))
        if len(self.picked) > 0:
            try:
                action_index = self.filtered_df['cid'].tolist().index(self.picked[-1])
                self.all_embedding[action_index, :, self.report_max_len:] = np.full_like(
                    self.all_embedding[action_index, :, self.report_max_len:], 0,
                    dtype=np.double)  # np.zeros_like(self.all_embedding[action_index])
            except Exception as ex:
                print(ex, action_index, self.current_id)
                raise ex
        self.previous_obs = self.all_embedding
        # print("Current_Id", self.current_id, type(self.all_embedding))
        # print(self.all_embedding.shape)
        return self.all_embedding.squeeze(1)

if __name__ == "__main__":
    #from stable_baselines3 import DQN
    #from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer, ReplayBufferSamples

    env = LTREnvV5(data_path="Data/TrainData/Bench_BLDS_Dataset.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=50, code_max_len=4096,
                   use_gpu=False, caching=True, window_size=500)
    #obs = env.reset()
    #env.step(2)
    # buff = get_replay_buffer(5000, env)
    # # buff = get_replay_buffer(5000, env, device="cuda:0",priority=True)
    # samples = buff.sample(30)
   # model = DQN("MlpPolicy", env, verbose=1, buffer_size=5000,)
    #model.learn(total_timesteps=100000)
    # Path("Models/RL").mkdir(parents=True, exist_ok=True)
    #model.save("Models/RL/DQN")
    # print("Loading Model ...")
    #model = DQN.load("Models/RL/DQN")
    #
    obs = env.reset()
   # for i in range(10000):
         #action, _states = model.predict(obs, deterministic=True)
         #obs, reward, done, info = env.step(action)
         #print("Reward: {}".format(reward))
    #     env.render()
         #if done:
            # obs = env.reset()
