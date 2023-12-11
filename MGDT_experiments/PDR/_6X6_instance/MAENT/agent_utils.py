from torch.distributions.categorical import Categorical


def select_action(p, cadidate, memory):
  
    dist = Categorical(p.squeeze())
    s = dist.sample()
    #print(cadidate.shape, p.shape,s,'agebe')
    if memory is not None: memory.logprobs.append(dist.log_prob(s))
    #print(cadidate.shape,s,cadidate[s]) #(50,) tensor(20, device='cuda:0') 400

    return cadidate[s], s

def select_action_i(p):

    dist = Categorical(p.squeeze())
    s = dist.sample()
    #print(cadidate.shape, p.shape,s,'agebe')

    #print(cadidate.shape,s,cadidate[s]) #(50,) tensor(20, device='cuda:0') 400

    return s
# evaluate the actions
def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


# select action method for test
def greedy_select_action(p, candidate):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    return action


# select action method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
