import numpy as np
import itertools
from math import comb
import math
import copy


def min_max_scale(x):
    x_new = (x - np.min(x))/(np.max(x)-np.min(x))
    return x_new

def mean_scale(x):
    x_new = (x - np.mean(x))/(np.max(x)-np.min(x))
    return x_new

def min_max_scale_ab(x, a, b):
    x_new = a + ( (x - np.min(x) ) * (b-a) ) / (np.max(x)-np.min(x))
    return x_new

def softmax_scale(x, tau):
    x_new = np.exp(x*tau)/sum(np.exp(x*tau))
    return x_new

def tau_function(x,a,b):
    tau = 2*a/(1+np.exp(-b*x)) - a +1
    return tau

def exponential_decay(a, b, c, N):
    # a, b: exponential decay parameter
    # N: number of samples 
    return (a-c) * (1-b) ** np.arange(N) + c

#Define the federated averaging function
def FedAvg(w,alpha):
    w_avg = copy.deepcopy(w[0])
    n_clients = len(w)
    alpha = alpha/np.sum(alpha)
    for l in w_avg.keys():
        w_avg[l] = w_avg[l].float() - w_avg[l]

    for l, layer in enumerate(w_avg.keys()): #for each layer
        for k in range(0,n_clients): #for each client
            w_avg[layer] += alpha[k]*w[k][layer]
    return w_avg

#Training function for the bandit algorithm
def train_bandit(n_rounds, n_local_epochs, clients, sample_frac, n_sampled, bandit_var, q_max,q_min):
    if(bandit_var):
        q_values = exponential_decay(q_max, 0.1, q_min, n_rounds)
        
    for i in range(len(clients)):
        _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
        print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")

    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)), int(sample_frac*len(clients)), replace=False)
        w_new_list = {}
        client_action_list = {}
        for i in idxs:
            action_list = list(itertools.combinations(set(range(len(clients))) - set([i]), n_sampled))
            if(not clients[i].stopped_early):
                action = clients[i].player3.sample_action(round)
                client_action_list[i] = action
                clients_sampled = action_list[action]
                #print(f"{i} sampled: {clients_sampled}")

                neighbour_stats = []
                for j in clients_sampled: #request models from n_sampled neighbours
                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    neighbour_stats.append((model_j.state_dict(), n_train, j))
                    assert(i != j)
                    clients[i].n_sampled[j] += 1

                w_avg = [neighbour_stats[k][0] for k in range(n_sampled)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][1] for k in range(n_sampled)]
                alpha.append(len(clients[i].train_set))

                w_new = FedAvg(w_avg, alpha)
                w_new_list[i] = w_new
        for i in idxs:
            clients[i].local_model.load_state_dict(w_new_list[i])
            val_loss, val_acc = clients[i].validate(clients[i].local_model, train_set=False)
            clients[i].rewards.append(val_acc/100) #reward between 0-1
            #print(i,client_action_list[i])
            clients[i].player3.update_policy(client_action_list[i], clients[i].rewards[round], round)
            if(bandit_var):
                clients[i].player3.q = q_values[round]
            
            _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)

            print(f"Round {round}, Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
            print("--"*10)
    return clients

#Training function for the PENS baseline algorithm
def train_pens(n_rounds, n_local_epochs, clients, sample_frac, n_sampled, top_m):
    
    for i in range(len(clients)):
        _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
        #print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)), int(sample_frac*len(clients)), replace=False)
        for i in idxs:
            if(not clients[i].stopped_early):
                neighbour_list = list(set(range(len(clients))) - set([i]))
                neighbour_sampled = np.random.choice(neighbour_list, n_sampled, replace=False)

                neighbour_stats = []
                for j in neighbour_sampled: #request models from n_sampled neighbours

                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    #validate model_j on client_i train set
                    train_loss_ij, _ = clients[i].validate(model_j,train_set = True)
                    neighbour_stats.append((model_j.state_dict(), train_loss_ij, n_train, j))

                    clients[i].n_sampled[j] += 1

                neighbour_stats.sort(key=lambda x:x[1])
                w_avg = [neighbour_stats[k][0] for k in range(top_m)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][2] for k in range(top_m)]
                alpha.append(len(clients[i].train_set))

                selected_neighbours = [neighbour_stats[k][3] for k in range(top_m)]

                for k in selected_neighbours:
                    clients[i].n_selected[k] += 1

                w_new = FedAvg(w_avg,alpha)
                clients[i].local_model.load_state_dict(w_new)
                _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
                
                print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
            
    return clients

#Training function for the DAC baseline algorithm
def train_dac(n_rounds, n_local_epochs, clients, sample_frac, n_sampled, tau, dac_var):
    
    for i in range(len(clients)):
        _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
        print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)), int(sample_frac*len(clients)), replace=False)
        w_new_list = {}
        ik_similarity_list = {}
        neighbour_idx_list = {}
        new_neighbours_list = {}
        for i in idxs:
            if(not clients[i].stopped_early):
                neighbour_list = list(range(len(clients)))
                if(round==0):
                    probas = np.ones(len(clients))/len(clients)
                    probas[i] = 0.0
                    probas = probas/np.sum(probas)
                else:
                    probas = clients[i].priors_norm
                neighbour_sampled = np.random.choice(neighbour_list, n_sampled, replace=False, p=probas)

                neighbour_stats = []
                for j in neighbour_sampled: #request models from n_sampled neighbours

                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    train_loss_ij, _ = clients[i].validate(model_j, train_set = True)
                    neighbour_stats.append((model_j.state_dict(), train_loss_ij, n_train, j))

                    clients[i].n_sampled[j] += 1

                w_avg = [neighbour_stats[k][0] for k in range(n_sampled)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][2] for k in range(n_sampled)]
                alpha.append(len(clients[i].train_set))
                
                ik_similarity = [1/(neighbour_stats[k][1]) for k in range(n_sampled)]
                ik_similarity_list[i] = ik_similarity

                neighbour_idx = [neighbour_stats[k][3] for k in range(n_sampled)]
                neighbour_idx_list[i] = neighbour_idx
                
                neighbour_list = np.arange(len(clients))
                new_neighbours = []
                for k in neighbour_idx:
                    new_neighbours += list(set(neighbour_list[clients[k].priors > 0]) - set(neighbour_list[clients[i].n_sampled > 0]) - set([i]))
                
                new_neighbours = np.unique(new_neighbours)
                new_neighbours_list[i] = new_neighbours
                
                w_new = FedAvg(w_avg,alpha)
                w_new_list[i] = w_new

        for i in idxs:
            for k in range(n_sampled):
                clients[i].priors[neighbour_idx_list[i][k]] = ik_similarity_list[i][k]

            for j in new_neighbours_list[i]:
                score_kj_array = np.zeros(n_sampled)
                ki_scores = []
                for k in range(n_sampled):
                    score_kj = clients[neighbour_idx_list[i][k]].priors[j]
                    score_kj_array[k] = score_kj
                    if(score_kj>0):
                        #save tuple (ki_score, k)
                        ki_scores.append( (clients[i].priors[neighbour_idx[k]], k ))
                ki_scores.sort(key=lambda x:x[0]) #sort tuple
                ki_max = ki_scores[-1][1] #choose k with max similarity to i
                clients[i].priors[j] = score_kj_array[ki_max]
            
            not_i_idx = np.arange(len(clients[i].priors))!=i
            if(dac_var):
                tau_new = tau_function(round,tau,0.2)
            else:
                tau_new = tau
                    
            clients[i].priors_norm[not_i_idx] = softmax_scale(clients[i].priors[not_i_idx], tau_new)

            clients[i].local_model.load_state_dict(w_new_list[i])
            _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
            print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")

    return clients
# Training function for the gossip baseline algorithm
def train_gossip(n_rounds, n_local_epochs, clients, sample_frac, n_sampled, is_pens=False):
    
    for i in range(len(clients)):
        clients[i].count = 0
        clients[i].stopped_early = False
        if(not is_pens):
            _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
            print(f"Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
        
    for round in range(n_rounds):
        idxs = np.random.choice(range(len(clients)),int(sample_frac*len(clients)),replace=False)
        w_new_list = {}
        for i in idxs:
            if(not clients[i].stopped_early):
                if(n_sampled > len(clients[i].neighbours)):
                    n_sampled_neighbours = len(clients[i].neighbours)
                else:
                    n_sampled_neighbours = n_sampled

                neighbour_sampled = np.random.choice(clients[i].neighbours, n_sampled_neighbours, replace=False)
                neighbour_stats = []
                for j in neighbour_sampled: #request models from n_sampled neighbours

                    model_j = clients[j].local_model
                    n_train = len(clients[j].train_set)
                    neighbour_stats.append((model_j.state_dict(), n_train, j))
                    clients[i].n_sampled[j] += 1

                w_avg = [neighbour_stats[k][0] for k in range(n_sampled_neighbours)]
                w_avg.append(clients[i].local_model.state_dict())

                alpha = [neighbour_stats[k][1] for k in range(n_sampled_neighbours)]
                alpha.append(len(clients[i].train_set))
                w_new = FedAvg(w_avg,alpha)
                w_new_list[i] = w_new
        for i in idxs:
            clients[i].local_model.load_state_dict(w_new_list[i])
            _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs)
            print(f"Round {round} | Client {i} | Train loss {np.round(train_loss,2)} | Val loss {np.round(val_loss,2)} | Val acc {np.round(val_acc,2)}")
            
    return clients