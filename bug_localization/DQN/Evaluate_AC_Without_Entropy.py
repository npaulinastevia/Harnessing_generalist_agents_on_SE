import glob
import random
import pickle
from tqdm import tqdm
import  json
import argparse
import torch
#from AC import to_one_hot
#from Evaluate_Random import calculate_top_k
from AC_Without_Entropy import PolicyModel as NewPolicyModel
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from Buffer import get_replay_buffer, get_priority_replay_buffer
import numpy as np
from Environment import LTREnvV2
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
#from stable_baselines3.common.buffers import ReplayBuffer
from Buffer import CustomBuffer
def calculate_top_k(source, target, counts):
    return
def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    current_directory = os.getcwd()
    parser.add_argument('--file_path', default="/project/def-m2nagapp/partha9/LTR/", help='File Path')
    parser.add_argument('--test_data_path', help='Test Data Path')
    parser.add_argument('--project_name',default='AspectJ',  help='Project Name')
    parser.add_argument('--model_path', help='Project Name')
    parser.add_argument('--result_path', help='Project Name')
    parser.add_argument('--run', default='1',type=str, help='Project Name')
    parser.add_argument('--algo', default='AC_WITHOUT_ENTROPY', type=str, help='Project Name')
    parser.add_argument('--non_original', default=1, type=int, help='Project Name')
    parser.add_argument('--reg_path', default='', type=str, help='Project Name')
    parser.add_argument('--estimate', default=False, type=bool, help='Project Name')
    parser.add_argument('--base', default='baseline', type=str, help='Project Name')
    parser.add_argument('--test_path', default='', type=str, help='Project Name')
    options = parser.parse_args()
    # file_path = "/project/def-m2nagapp/partha9/LTR/"
    # test_data_path = "Data/TestData/AspectJ_test.csv"
    # project_name = "AspectJ"
    if options.base=='baseline':
        test_file=options.test_path
    else:
        test_file=options.test_path
    file_path = os.path.join(current_directory, options.run+'_LTR')
    test_data_path = test_file#os.path.join(current_directory, options.project_name+'_test.csv')#os.path.join(current_directory, options.project_name+'_test.csv')
    if options.estimate:
        test_data_path = os.path.join(current_directory, options.project_name + '.csv')
    project_name = options.project_name
    project_namee = project_name
    print(project_namee,project_name)
    model_path = os.path.join(file_path, 'Models',options.algo,project_namee)
    final_directory = os.path.join(current_directory, options.algo+'_'+project_namee+'_'+options.run)
    os.makedirs(final_directory, exist_ok=True)
    results_test=os.path.join(final_directory,options.base+'results_testn.txt')
    box_test = os.path.join(final_directory, options.base + 'box_testn.txt')
    finalr=os.path.join(final_directory,options.base+'finalrn.txt')
    fi = open(finalr, 'a+')
    fi.write('mrr,actualrank' + '\n')
    fi.close()
    fi = open(results_test, 'a+')
    fi.write('algo,project_name,reward,time,episode,steps,mrr,map,actualr,top1,top5,top10'+ '\n')
    fi.close()
    fi = open(box_test, 'a+')
    fi.write('algo,project_name,bug_id,mrr,map,top1,top5,top10' + '\n')
    fi.close()
    result_path = final_directory
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using {}".format(dev))
    #metric = options.run[1:]
    if len(options.run)>1:
        metric=options.run[1:]
    else:
        metric=''
    if options.non_original==0:
        non_original = False
    else:
        non_original = True
    reg_path=options.reg_path
    mpath='/scratch/f/foutsekh/nstevia/bug_localization/micro_codebert'

    env = LTREnvV2(data_path=test_data_path, model_path=mpath,#data_path=file_path + test_data_path
                   tokenizer_path=mpath, action_space_dim=31, report_count=None, max_len=512,
                   use_gpu=True, caching=True, file_path=file_path, project_list=[project_name], test_env=True,estimate=options.estimate,metric=metric,non_original=non_original,reg_path=reg_path)

    model = NewPolicyModel(env=env)
    print(model_path,'model_pathhhh')
    for f in sorted(glob.glob(os.path.join(model_path,'*.pt'))):#SWT_New_AC_Entropy
        if f.startswith(os.path.join(model_path,project_namee+'_'+'New_AC_Entropy_policy_model')):
            model_f=f

    state_dict = torch.load( model_f)#(file_path + model_path)
    model.load_state_dict(state_dict=state_dict)
    model = model.to(dev)
    all_rr = []
    counts = None
    import time
    start = time.time()
    st = 0
    total_e = 0
    all_map=[]
    map_per_r=[]
    final_top=[0,0,0]
    for _ in tqdm(range(env.suppoerted_len)):
        all_rr.append(-100)
        done = False
        picked = []
        hidden = [torch.zeros([1, 1, model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, model.lstm_hidden_space]).to(dev)]
        prev_obs = env.reset()
        total_e += 1
        rew_all=0
        irr=False
        while not done:
            st+=1
            prev_actions = to_one_hot(picked, max_size=env.action_space.n)
            prev_actions = torch.from_numpy(prev_actions).to(dev).type(torch.float)
            prev_obs = torch.from_numpy(np.expand_dims(prev_obs, axis=0)).float().to(dev)
            hidden = [item.to(dev).type(torch.float) for item in hidden]
            with torch.no_grad():
                action, hidden = model(x=prev_obs, actions=prev_actions, hidden=hidden)
            action = torch.distributions.Categorical(action).sample()
            action = int(action[0][0].detach().cpu().numpy())
            prev_obs, reward, done, info, rr,map = env.step(action, return_rr=True) #algo,project_name,reward,time,episode,steps,mrr,map,actualr,top1,top5,top10

            if options.estimate:
                if env.current_id in env.irr:
                    map,rr=0,0
                    irr=True

            rew_all+=reward
            picked.append(action)
            if all_rr[-1] < rr:
                all_rr[-1] = rr
            counts = calculate_top_k(source=env.picked, target=env.match_id, counts=counts)
        real_fix=0
        precision_at_k=[]
        print('next')
        print(env.picked)
        print(env.match_id)
        print('nextnnnn')
        for kk in range( len(env.picked)):
            if env.picked[kk] in env.match_id:
                real_fix+=1
            precision_at_k.append(real_fix/(kk+1))
            all_map.append(precision_at_k)
        map_per_r.append(np.mean(np.array(precision_at_k)))
        temp=np.array(all_rr)
        temp=temp[temp > 0]
        top=[]
        if irr:
            top=[0,0,0]
        else:
            for u in [1,5,10]:
                cc=0

                for uu in range(u):
                   
                    if len(env.picked) >= uu + 1:
                        if env.picked[uu] in env.match_id:
                            cc+=1
                top.append(cc)
                
        #algo,project_name,bug_id,mrr,map,top1,top5,top10'
        ret=all_rr[-1] if all_rr[-1]>0 else -1
        fi = open(box_test, 'a+')
        fi.write(
            options.algo + "," + project_name + "," + str(total_e) + "," + str(ret) + "," + str(map_per_r[-1]) + "," + str(top[0]) +
            "," + str(top[1]) + "," + str(top[2])  + '\n')
        fi.close()
        final_top[0],final_top[1],final_top[2]=final_top[0]+top[0],final_top[1]+top[1],final_top[2]+top[2]
        fi = open(results_test, 'a+')
        fi.write(
            options.algo + "," + project_name + "," + str(rew_all) + "," + str(-start + time.time()) + "," + str(
                    total_e) + "," + str(st) +"," + str(temp.mean()) + "," + str(np.mean(np.array(map_per_r)))+ "," +str(final_top[0]/env.suppoerted_len)+ "," +str(final_top[1]/env.suppoerted_len)+ "," +str(final_top[2]/env.suppoerted_len) +'\n')
        fi.close()
    all_rr = np.array(all_rr)
    all_rr = all_rr[all_rr > 0]
    mean_rr = all_rr.mean() if len(all_rr)>0 else 0
    actual_rank = 1.0/all_rr
    fi = open(finalr, 'a+')
    fi.write(str(mean_rr) +"," +str(actual_rank)+"," +str(all_map)+"," +str(all_rr)+  '\n')
    fi.close()



    Path(result_path).mkdir(exist_ok=True,parents=True)
    json.dump({"mrr": mean_rr}, open(result_path + "_mrr.json", "w"))
    np.save(result_path + "_ranks.npy", actual_rank)
    #plt.figure(figsize=(500, 500))
    #plt.hist(1.0/all_rr, bins=30)
    #plt.savefig(result_path + "_histogram.eps", format='eps', dpi=50)
    #plt.figure(figsize=(500, 500))
    #plt.boxplot(1.0/all_rr)
    #plt.savefig(result_path + "_boxplot.eps", format='eps', dpi=50)
