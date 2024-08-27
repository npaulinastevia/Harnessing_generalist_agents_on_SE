import glob
#import psutil
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from Buffer import CustomBuffer
#from DQN import train_dqn_epsilon, to_one_hot
from Environment import LTREnvV2
import torch.nn.functional as F
import numpy as np
import pickle

def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)

class ValueModel(nn.Module):
    def __init__(self, env):
        super(ValueModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.lstm_hidden_space = 256

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        linear_input_size =convw * convh * 32 + env.action_space.n
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
        self.lin_layer2 = nn.Linear(self.lstm_hidden_space, 1)

    def forward(self, x, actions, hidden=None):
        x = x.unsqueeze(1) if x.dim() == 3 else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.concat([x.unsqueeze(1), actions.unsqueeze(1) if actions.dim() != 3 else actions], axis=2)
        x, (new_h, new_c) = self.lstm(x, (hidden[0], hidden[1]))
        x = self.lin_layer2(x)
        return x, [new_h, new_c]


class PolicyModel(nn.Module):
    def __init__(self, env):
        super(PolicyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.lstm_hidden_space = 256

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        linear_input_size = convw * convh * 32 + env.action_space.n
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
        self.lin_layer2 = nn.Linear(self.lstm_hidden_space, env.action_space.n)

    def forward(self, x, actions, hidden=None):
        x = x.unsqueeze(1) if x.dim() == 3 else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        #x = F.relu(self.lin_layer1(x))
        x = torch.concat([x.unsqueeze(1), actions.unsqueeze(1) if actions.dim() != 3 else actions], axis=2)
        x, (new_h, new_c) = self.lstm(x, (hidden[0], hidden[1]))
        x = self.lin_layer2(x)
        # return torch.softmax((x * actions), dim=-1), [new_h, new_c]
        x = torch.softmax(x, dim=-1) * actions
        x = x / x.sum()
        return x, [new_h, new_c]


def a2c_step(policy_net, optimizer_policy, optimizer_value, states, advantages, batch_picked, batch_hidden, lambda_val=1):
    """update critic"""
    value_loss = advantages.pow(2).mean()
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    probs, _ = policy_net(states, actions=batch_picked, hidden=batch_hidden)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    policy_loss = -dist.log_prob(action) * advantages.detach()  + lambda_val * dist.entropy()
    policy_loss = policy_loss.mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, done, states, next_states, gamma, device, value_model, batch_hidden_value, batch_picked):
    rewards, masks, states, next_states = rewards.to(device), done.to(device).type(torch.float), states.to(device).type(
        torch.float), next_states.to(device).type(torch.float)
    advantages = rewards + (1.0 - masks) * gamma * value_model(next_states, batch_picked, batch_hidden_value)[0].detach() - value_model(states, batch_picked, batch_hidden_value)[0]
    return advantages


def update_params(samples, value_net, policy_net, policy_optimizer, value_optimizer, gamma, device):
    state, action, reward, next_state, done, info = samples
    batch_hidden = torch.tensor(np.array(
        [np.stack([np.array(item['hidden'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden'][1]) for item in info], axis=2)[0]])).to(device)
    batch_hidden_value = torch.tensor(np.array(
        [np.stack([np.array(item['hidden_value'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden_value'][1]) for item in info], axis=2)[0]])).to(device)
    batch_picked = torch.tensor(np.array(
        [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info])).to(device).type(
        torch.float)

    """get advantage estimation from the trajectories"""
    advantages = estimate_advantages(reward, done, state, next_state, gamma, device, value_net, batch_hidden_value, batch_picked)

    """perform TRPO update"""
    a2c_step(policy_net, policy_optimizer, value_optimizer, state.type(torch.float).to(device), advantages,
             batch_picked, batch_hidden)


def train_actor_critic(total_time_step, sample_size, project_name,options,results_train, save_frequency=30):
    import time
    start=time.time()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy_model = PolicyModel(env=env)
    value_model = ValueModel(env=env)
    if prev_policy_model_path is not None:
        state_dict = torch.load(prev_policy_model_path)
        policy_model.load_state_dict(state_dict=state_dict)
    if prev_value_model_path is not None:
        state_dict = torch.load(prev_value_model_path)
        value_model.load_state_dict(state_dict=state_dict)
    policy_model = policy_model.to(dev)
    value_model = value_model.to(dev)
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=0.01)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=0.01)

    OBSERVATION_SHAPE = (31, 1025)  # (84, 84) (20, 20)
    PATCH_SHAPE = (5, 5)  # The size of tensor a (16) must match the size of tensor b (36) at non-singleton dimension 2
    NUM_ACTIONS = 31  # 18 Maximum number of actions in the full dataset.
    # rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
    NUM_REWARDS = 3
    RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset
    state_dim, self.act_dim, self.action_range = None,31,None
    target_entropy = -act_dim
    model = MultiGameDecisionTransformer(
        img_size=OBSERVATION_SHAPE,
        patch_size=PATCH_SHAPE,
        num_actions=NUM_ACTIONS,
        num_rewards=NUM_REWARDS,
        return_range=RETURN_RANGE,
        d_model=1280,
        num_layers=10,
        act_dim=act_dim,
        dropout_rate=0.1,
        predict_reward=True,
        single_return_token=True,
        conv_dim=256,
        stochastic_policy=False,
        max_ep_len=MAX_EPISODE_LEN,
        eval_context_length=options.eval_context_length,
        init_temperature=options.init_temperature,
        state_dim=state_dim,
        target_entropy=target_entropy,
        parser=parser,
    )

    # --- Load pretrained weights
    self.parser = parser
    from load_pretrained import load_jax_weights
    model_params, model_state = pickle.load(open(
        "/scratch/f/foutsekh/nstevia/Harnessing_generalist_agents_on_SE/MGDT_experiments/bug_localization/models/checkpoint_38274228.pkl",
        "rb"))
    load_jax_weights(model, model_params)
    model = model.to(dev)
    pbar = tqdm(range(total_time_step))
    episode_len_array = []
    episode_reward = []
    total_e=0
    for e in pbar:
        done = False
        prev_obs = env.reset()
        hidden = [torch.zeros([1, 1, policy_model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, policy_model.lstm_hidden_space]).to(dev)]
        hidden_value = [torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev)]
        picked = []
        reward_array = []
        # pbar.set_description("Avg. reward {} Avg. episode {} Mem: {}".format(np.array(episode_reward).mean(),
        #                                                              np.array(episode_len_array).mean(),
        #                                                                      psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
        pbar.set_description("Avg. reward {} Avg. episode {}".format(np.array(episode_reward).mean(),
                                                                             np.array(episode_len_array).mean()))
        episode_len = 0
        total_e+=1
        rew_all=0
        while not done:

            episode_len += 1
            prev_obs = torch.Tensor(prev_obs).to(dev)
            prev_obs = prev_obs.unsqueeze(0)
            temp_action = torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).to(
                dev).type(torch.float)
            with torch.no_grad():
                action, temp_hidden = policy_model(prev_obs, actions=temp_action, hidden=hidden)
                _, temp_hidden_value = policy_model(prev_obs, actions=temp_action, hidden=hidden_value)
            action = torch.distributions.Categorical(action).sample()
            action = int(action[0][0].cpu().numpy())
            picked.append(action)
            obs, reward, done, info = env.step(action)

            rew_all+=reward
            reward_array.append(reward)
            info['hidden'] = [item.cpu().numpy() for item in hidden]
            info['picked'] = picked
            info['hidden'] = [item.cpu().numpy() for item in hidden]
            info['hidden_value'] = [item.cpu().numpy() for item in hidden_value]
            hidden = temp_hidden
            hidden_value = temp_hidden_value
            buffer.add(prev_obs.cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
        import time
        fi = open(results_train, 'a+')
        fi.write(options.algo+","+project_name+","+str(rew_all)+","+str(-start+time.time())+","+str(total_e)+","+str(episode_len) + '\n')
        fi.close()
        if len(buffer) > 50:
            samples = buffer.sample(sample_size)
            update_params(samples=samples, value_net=value_model, policy_net=policy_model,
                          policy_optimizer=optimizer_policy, value_optimizer=optimizer_value, gamma=0.99, device=dev)
            episode_reward.append(np.array(reward_array).sum())
            episode_len_array.append(episode_len)
        if e % save_frequency == 0:
            save_num = e / save_frequency
            if os.path.isfile(os.path.join(save_path , "{}_New_AC_Entropy_policy_model_{}.pt".format(project_name, save_num - 1))):
                os.remove(os.path.join(save_path , "{}_New_AC_Entropy_policy_model_{}.pt".format(project_name, save_num - 1)))
            if os.path.isfile(os.path.join(save_path , "{}_New_AC_Entropy_value_model_{}.pt".format(project_name, save_num - 1))):
                os.remove(os.path.join(save_path , "{}_New_AC_Entropy_value_model_{}.pt".format(project_name, save_num - 1)))
            if os.path.isfile(os.path.join(save_path , "{}_New_AC_Entropy_Episode_Reward.pickle".format(project_name))):
                os.remove(os.path.join(save_path , "{}_New_AC_Entropy_Episode_Reward.pickle".format(project_name)))
            if os.path.isfile(os.path.join(save_path , "{}_New_AC_Entropy_Episode_Length.pickle".format(project_name))):
                os.remove(os.path.join(save_path , "{}_New_AC_Entropy_Episode_Length.pickle".format(project_name)))

            torch.save(policy_model.state_dict(), os.path.join(save_path , "{}_New_AC_Entropy_policy_model_{}.pt".format(project_name, save_num)))
            torch.save(value_model.state_dict(), os.path.join(save_path , "{}_New_AC_Entropy_value_model_{}.pt".format(project_name, save_num)))

            with open(os.path.join(save_path ,"{}_New_AC_Entropy_Episode_Reward.pickle".format(project_name)), "wb") as f:
                pickle.dump(episode_reward, f)

            with open(os.path.join(save_path , "{}_New_AC_Entropy_Episode_Length.pickle".format(project_name)), "wb") as f:
                pickle.dump(episode_len_array, f)
    return policy_model, value_model


if __name__ == "__main__":
    current_directory = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default=r"C:\Users\phili\Downloads\10428077\Replication\LTR\\", help='File Path') #r"C:\Users\phili\Downloads\10428077\Replication\LTR\\
    parser.add_argument('--cache_path', default=r"C:\Users\phili\Downloads\10428077\Replication\.buffer_cache_ac", help='Cache Path')
    parser.add_argument('--prev_policy_model_path', default=None, help='Trained Policy Path')
    parser.add_argument('--prev_value_model_path', default=None, help='Trained Value Path')
    parser.add_argument('--train_data_path',default=r'C:\Users\phili\Downloads\10428077\Replication\org.aspectj\AspectJ.csv', help='Training Data Path')
    parser.add_argument('--save_path',default=r'C:\Users\phili\Downloads\10428077\Replication\LTR\Models\AC\Entropy\AspectJ\\', help='Save Path')
    parser.add_argument('--project_name',default='AspectJ',  help='Project Name')
    parser.add_argument('--algo', default='AC_WITHOUT_ENTROPY', type=str, help='Project Name')
    parser.add_argument('--run', default='1',type=str, help='Project Name')
    parser.add_argument('--finetune', default='0_f', type=str, help='Project Name')
    parser.add_argument('--non_original', default=0, type=int, help='Project Name')
    parser.add_argument('--reg_path', default='', type=str, help='Project Name')
    parser.add_argument('--timestepsR', default=0, type=int, help='Project Name')
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
    options = parser.parse_args()
    file_path = os.path.join(current_directory, options.run+'_LTR')
    cache_path = os.path.join(file_path, '.buffer_cache_ac')
    prev_policy_model_path = options.prev_policy_model_path
    prev_value_model_path = options.prev_value_model_path
    train_data_path = os.path.join(current_directory, options.train_data_path)# r'C:\Users\phili\Downloads\10428077\Replication\LTR\Models\AC\Entropy\AspectJ\\'
    #train_data_path='/scratch/f/foutsekh/nstevia/bug_localization/RL_Model/bug_loc_experiments/AspectJ_train_for_mining_before.csv'
    project_name = options.project_name
    save_path = os.path.join(file_path, 'Models',options.algo,options.project_name)

    #current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, options.algo+'_'+options.project_name+'_'+options.run)
    os.makedirs(final_directory, exist_ok=True)
    results_train=os.path.join(final_directory,'results_train.txt')
    fi = open(results_train, 'a+')
    fi.write('algo,project_name,reward,time,episode,steps'+ '\n')
    fi.close()
    Path(file_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True,exist_ok=True)
    #used to be data_path=file_path + train_data_path,
    mpath='/scratch/f/foutsekh/nstevia/bug_localization/micro_codebert'
    if len(options.run)>1:
        metric=options.run[1:]
    else:
        metric=''
    if options.non_original==0:
        non_original = False
    else:
        non_original = True
    print(non_original)

    reg_path = options.reg_path
    env = LTREnvV2(data_path=train_data_path, model_path=mpath,
                   tokenizer_path=mpath, action_space_dim=31, report_count=100, max_len=512,
                   use_gpu=True, caching=True, file_path=file_path, project_list=[project_name],metric=metric,non_original=non_original,reg_path=reg_path)
    for f in sorted(glob.glob(os.path.join(save_path,'*.pt'))):
         if f.startswith(os.path.join(save_path,project_name+'_'+'New_AC_Entropy_policy_model')):
             prev_policy_model_path=f
         elif f.startswith(os.path.join(save_path,project_name+'_'+'New_AC_Entropy_value_model')):
             prev_value_model_path=f
    obs = env.reset()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    buffer = CustomBuffer(6000, cache_path=cache_path)
    policy, value = train_actor_critic(total_time_step=((7500*int(options.finetune[0]))/100), sample_size=128, project_name=project_name,options=options,results_train=results_train)
