import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
from pathlib import Path
from options import args_parser
import yaml

from models import *
from train_funs import *
from utils import *

def create_test_data_covariate(cluster_id,rot_deg, dataset_name):
    if(dataset_name=='cifar10'):
        rot_transform = transforms.RandomRotation(degrees=(rot_deg[cluster_id],rot_deg[cluster_id]))
        trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),rot_transform])
        test_dataset = torchvision.datasets.CIFAR10('./data/', train=False, download=True, transform=trans_cifar_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
    elif(dataset_name=='cifar100'):
        rot_transform = transforms.RandomRotation(degrees=(rot_deg[cluster_id],rot_deg[cluster_id]))
        trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),rot_transform])
        test_dataset = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=trans_cifar_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
    elif(dataset_name=='fashion-mnist'):
        rot_transform = transforms.RandomRotation(degrees=(rot_deg[cluster_id],rot_deg[cluster_id]))
        trans_fashion_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),rot_transform])

        test_dataset = torchvision.datasets.FashionMNIST('./data/', train=False, download=True, transform=trans_fashion_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
    return test_loader

def create_test_data_label(dataset):
    if(dataset=='cifar10'):
        trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = torchvision.datasets.CIFAR10('./data/', train=False, download=True, transform=trans_cifar_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
    elif(dataset=='cifar100'):
        trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=trans_cifar_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
    elif(dataset == 'fashion-mnist'):
        trans_fashion_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        test_dataset = torchvision.datasets.FashionMNIST('./data/', train=False, download=True, transform=trans_fashion_test)
        test_loader = DataLoader(test_dataset, batch_size=1)
    
    return test_loader

if __name__ == '__main__':   
    args = args_parser()
    
    
    if(args.seed == None):
        seed = random.randint(1,10000)
    else:
        seed = args.seed
        
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create experiment directory
    if(args.DAC):
        method_name = f'DAC_{args.tau}'
    elif(args.DAC_var):
        method_name = f'DAC_var_{args.tau}'
    elif(args.oracle):
        method_name = 'Oracle'
    elif(args.pens):
        method_name = 'PENS'
    elif(args.random):
        method_name = 'Random'
    elif(args.bandit):
        method_name = f'Bandit_{args.q}'
    elif(args.bandit_var):
        method_name = f'Bandit_var_{args.q}_{args.q_min}'
    
    count = 0
    experiment_dir = f"./save/{method_name}_{args.iid}_{seed}_{count}"
    while os.path.exists(experiment_dir):
        count=count+1
        experiment_dir = f"./save/{method_name}_{args.iid}_{seed}_{count}"
        
    os.mkdir(experiment_dir)
    
    with open(Path(experiment_dir) / "args.yml", "w") as f:
        yaml.dump(args, f)
        
    filename = 'results'
    filexist = os.path.isfile(experiment_dir+'/'+filename) 
    if(not filexist):
        with open(experiment_dir+'/'+filename,'a') as f1:

            f1.write('n_rounds;n_rouns_pens;num_clients;local_ep;bs;lr;n_clusters;pens;DAC;DAC_var;oracle;random;bandit;bandit_var;top_m;n_sampled;n_data_train;n_data_val;tau;q;q_min;shift;test_acc0;test_acc1;test_acc2;test_acc3;test_acc4;dataset')

            f1.write('\n')
            
    # Set the device
    cuda_no = str(args.gpu)
    device = torch.device("cuda:"+cuda_no)
    criterion = nn.NLLLoss()
    
    num_clients = args.num_clients
    
    # Load the dataset and split among users
    if(args.dataset=='fashion-mnist'):
        num_classes = 10
        trans_fashion = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST('./data/', train=True, download=True, transform=trans_fashion)

        #trans_fashion_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #test_dataset = torchvision.datasets.FashionMNIST('./data/', train=False, download=True, transform=trans_fashion)
    elif(args.dataset=='cifar10'):
        num_classes = 10
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10('./data/', train=True, download=True, transform=trans_cifar)

        #trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, 0.5, 0.5))])
        #test_dataset = torchvision.datasets.CIFAR10('./data/', train=False, download=True, transform=trans_cifar_test)
    elif(args.dataset=='cifar100'):
        num_classes = 100
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=trans_cifar)

        #trans_cifar_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, 0.5, 0.5))])
        #test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=trans_cifar_test)
        
    
    #Assign data samples to clients
    
    if(args.iid == 'label'):
        if(args.dataset == 'fashion-mnist'):
            print("Error: Dataset Fahsion-MNIST not yet implemented for label shift. Try covariate-shift or change to CIFAR-10")
            exit()
        if(args.dataset=='cifar10'):
            dict_users, dict_users_val = sample_cifargroups(train_dataset, num_clients, args.n_data_train, args.n_data_val) 
            cluster_idx = np.zeros(num_clients,dtype=int)
            cluster_idx[0:int(0.4*num_clients)] = 1 #black (vehicles)
        elif(args.dataset=='cifar100'):
            dict_users, dict_users_val = sample_cifar100_groups(train_dataset, num_clients, args.n_data_train, args.n_data_val) 
            
    elif(args.iid == 'label2'):
        if(args.dataset == 'fashion-mnist'):
            cluster_idx = np.zeros(num_clients,dtype=int)
            dict_users, dict_users_val = cifar_noniid2(train_dataset, num_clients, 1.0, args.n_data_train, args.n_data_val, overlap=True)
        elif(args.dataset=='cifar10'):
            cluster_idx = np.zeros(num_clients,dtype=int)
            dict_users, dict_users_val = cifar_noniid2(train_dataset, num_clients, 1.0, args.n_data_train, args.n_data_val, overlap=True)
            
    elif(args.iid == 'label3'):
        cluster_idx = np.zeros(num_clients,dtype=int)
        cluster_idx[20:40] = 1
        cluster_idx[40:60] = 2
        cluster_idx[60:80] = 3
        cluster_idx[80:100] = 4
        if(args.dataset == 'fashion-mnist'):
            dict_users, dict_users_val, cluster_labels = cifar_noniid_5k(train_dataset, num_clients, args.n_data_train, args.n_data_val)
        elif(args.dataset=='cifar10'):
            dict_users, dict_users_val, cluster_labels = cifar_noniid_5k(train_dataset, num_clients, args.n_data_train, args.n_data_val)
        
    elif('covariate' in args.iid):
        dict_users, dict_users_val = sample_labels_iid(train_dataset, num_clients, args.n_data_train, args.n_data_val)
        cluster_idx = np.zeros(num_clients,dtype='int')
        cluster_list = []
        if(args.n_clusters==4):
            #hard coded as of now
            if args.iid == 'covariate2':
                rot_deg = [0,90,180,270] #cluster rotations
                sizes = [0.25, 0.5, 0.75] #cluster sizes
            
            elif args.iid == 'covariate1':
                sizes = [0.7, 0.9, 0.95] #cluster sizes
                rot_deg = [0, 180, 10, 350] #cluster rotations

            #hard-coded as of now
            cluster_idx[0:round(sizes[0]*num_clients)] = np.zeros(round(sizes[0]*num_clients),dtype='int')
            cluster_idx[round(sizes[0]*num_clients):round(sizes[1]*num_clients)] = 1*np.ones(round((sizes[1]-sizes[0])*num_clients),dtype='int')
            cluster_idx[round(sizes[1]*num_clients):round(sizes[2]*num_clients)] = 2*np.ones(round((sizes[2]-sizes[1])*num_clients),dtype='int')
            cluster_idx[round(sizes[2]*num_clients):] = 3*np.ones(round((1-sizes[2])*num_clients),dtype='int')

            cluster_0 = np.where(cluster_idx==0)[0]
            cluster_1 = np.where(cluster_idx==1)[0]
            cluster_2 = np.where(cluster_idx==2)[0]
            cluster_3 = np.where(cluster_idx==3)[0]
        elif(args.n_clusters == 2):
            rot_deg = [0, 180]
            cluster_0 = np.where(cluster_idx==0)[0]
            cluster_1 = np.where(cluster_idx==1)[0]
                    

    clients = []
    #Start the training process
    for idx in range(args.num_clients):
        if((args.iid == 'label') or (args.iid == 'label2') or (args.iid == 'label3')):
            rot_deg_i = 0
            
        elif('covariate' in args.iid):
            rot_deg_i = rot_deg[cluster_idx[idx]]
            cluster_list.append(rot_deg_i)
        num_actions = comb(num_clients,args.n_sampled)
        print(f"Creating clients... {idx}",end="\r")
        client = ClientUpdate(train_dataset, dict_users[idx], 
                              dict_users_val[idx], args.dataset, criterion, args.lr, device, args.bs, args.iid,
                              rot_deg_i, args.num_clients, num_actions, args.n_sampled, args.q, idx)
        clients.append(client)
                    
    sample_frac = 1.0
    if(args.pens):
        clients = train_pens(args.n_rounds_pens, args.local_ep, clients, sample_frac, args.n_sampled, args.top_m)
                    
        for i in range(len(clients)):
            expected_samples = (args.top_m/args.num_clients) * args.n_rounds_pens
            clients[i].neighbours = np.where(clients[i].n_selected > expected_samples)[0]
        
        clients = train_gossip(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.pens) 
        
    elif(args.DAC or args.DAC_var): 
        clients = train_dac(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.tau, args.DAC_var)
    
    elif(args.random):
        for i in range(len(clients)):
            clients[i].neighbours = list(set(np.arange(num_clients))-set([i]))
        clients = train_gossip(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.pens)
    
    elif(args.oracle):
        if(args.iid == 'label2'):
            label2_list = np.array(list(itertools.product(range(10),repeat=2)))[0:num_clients]
        for i in range(len(clients)):
            if('covariate' in args.iid):
                #here we choose neighbours using an oracle
                clients[i].neighbours = list(set(np.where(cluster_idx==cluster_idx[i])[0])-set([i]))
            elif(args.iid == 'label'):
                if(i<40):
                    clients[i].neighbours = list(set(np.arange(0,40))-set([i]))
                else:
                    clients[i].neighbours = list(set(np.arange(40,100))-set([i]))
            elif(args.iid == 'label2'):
                target_i0 = label2_list[i][0]
                target_i1 = label2_list[i][1]
                
                idxs0 = np.where(label2_list==target_i0)[0]
                idxs1 = np.where(label2_list==target_i1)[0]
                
                clients[i].neighbours = list(set(np.union1d(idxs0,idxs1))-set([i]))
                print(clients[i].neighbours)
                
            elif(args.iid == 'label3'):
                if(i in np.arange(0,20)):
                    clients[i].neighbours = list(set(np.arange(0,20))-set([i]))
                elif(i in np.arange(20,40)):
                    clients[i].neighbours = list(set(np.arange(20,40))-set([i]))
                elif(i in np.arange(40,60)):
                    clients[i].neighbours = list(set(np.arange(40,60))-set([i]))
                elif(i in np.arange(60,80)):
                    clients[i].neighbours = list(set(np.arange(60,80))-set([i]))
                elif(i in np.arange(80,100)):
                    clients[i].neighbours = list(set(np.arange(80,100))-set([i]))
                
                
        clients = train_gossip(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.pens)
        
    elif(args.bandit or args.bandit_var):
        clients = train_bandit(args.n_rounds, args.local_ep, clients, sample_frac, args.n_sampled, args.bandit_var, args.q, args.q_min)
     
    
    client_heatmap = np.zeros((len(clients),len(clients)))
    for j in range(len(clients)):
        client_heatmap[:,j] = clients[j].n_sampled
    
    #Plot the heatmap of client communication
    plt.figure(figsize=(12,8))
    sns.heatmap(client_heatmap)
    plt.savefig(f"{experiment_dir}/heatmap_{method_name}_nsampled_{args.n_sampled}_{args.dataset}_{args.iid}.png")
    
    if(args.bandit or args.bandit_var):
        plt.figure(figsize=(12,8))
        sns.set_theme()
        sns.lineplot(x=range(args.n_rounds),y=clients[0].rewards)
        sns.lineplot(x=range(args.n_rounds),y=clients[-1].rewards)
        plt.legend(['0',f'{len(clients)}'],fontsize=13)
        plt.xlabel('Round')
        plt.ylabel('Reward')
        plt.xticks(size=13);
        plt.yticks(size=13);
        plt.savefig(f"{experiment_dir}/rewards.png")
        
        
        with open(f"{experiment_dir}/rewards_vals1", "w") as f:
            for s in clients[0].rewards:
                f.write(str(s) +"\n")
                
        with open(f"{experiment_dir}/rewards_vals2", "w") as f:
            for s in clients[-1].rewards:
                f.write(str(s) +"\n")
        
    
    plt.figure(figsize=(12,8))
    sns.set_theme()
    plt.title('Train loss')
    col = ['r','k','b','magenta','cyan']
    for i in range(len(clients)):
        plt.plot(clients[i].train_loss_list,col[cluster_idx[i]],alpha=0.3)
        
    plt.savefig(f"{experiment_dir}/train_loss.png")
    
    plt.figure(figsize=(12,8))
    sns.set_theme()
    plt.title('Val loss')
    for i in range(len(clients)):
        plt.plot(clients[i].val_loss_list,col[cluster_idx[i]],alpha=0.3)
        
    plt.savefig(f"{experiment_dir}/val_loss.png")
    
    plt.figure(figsize=(12,8))
    sns.set_theme()
    plt.title('Val acc')
    for i in range(len(clients)):
        plt.plot(clients[i].val_acc_list,col[cluster_idx[i]],alpha=0.3)
        
    plt.savefig(f"{experiment_dir}/val_acc.png")
    
    # Evaluate the client models on test data
    acc_list0 = []
    acc_list1 = []
    acc_list2 = []
    acc_list3 = []
    acc_list4 = []

    if('covariate' in args.iid):
        test_loaders = []
        for nc in range(args.n_clusters):
            test_loader = create_test_data_covariate(nc,rot_deg,args.dataset)
            test_loaders.append(test_loader)
            
        for i in range(num_clients):
            if(i in cluster_0):
                _, acc = test(clients[i].best_model, criterion, test_loaders[0], device,num_classes)
                acc_list0.append(acc)
            elif(i in cluster_1):
                _, acc = test(clients[i].best_model, criterion, test_loaders[1], device,num_classes)
                acc_list1.append(acc)
            elif(i in cluster_2):
                _, acc = test(clients[i].best_model, criterion, test_loaders[2], device,num_classes)
                acc_list2.append(acc)
            elif(i in cluster_3):
                _, acc = test(clients[i].best_model, criterion, test_loaders[3], device,num_classes)
                acc_list3.append(acc)
    elif(args.iid == 'label'):
        if(args.dataset == 'cifar10'):
            test_loader = create_test_data_label(args.dataset)
        
            for i in range(args.num_clients):
                if(i<int(0.4*args.num_clients)):
                    group_labels = np.array([0,1,8,9]) #vehicles
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group_labels,10,args.dataset)
                    acc_list0.append(acc)
                else:
                    group_labels = np.array([2,3,4,5,6,7]) #animals
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group_labels,10,args.dataset)
                    acc_list1.append(acc)
        elif(args.dataset=='cifar100'):
            test_loader = create_test_data_label(args.dataset)
            group0 = np.array([0,1,7,8,11,12,13,14,15,16]) #animals
            group1 = np.array([2,4,17]) #trees and plants
            group2 = np.array([3,5,6]) #things
            group3 = np.array([9,10]) #nature and manmade buildings
            group4 = np.array([18,19]) #vehicles
            for i in range(args.num_clients):
                print(f"Testing client {i}")
                if(i<int(args.num_clients*(10/20))): #group0
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group0,100,args.dataset)
                    acc_list0.append(acc)
                elif(i<int(args.num_clients*(13/20))): #group1
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group1,100,args.dataset)
                    acc_list1.append(acc)
                elif(i<int(args.num_clients*(16/20))): #group2
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group2,100,args.dataset)
                    acc_list2.append(acc)
                elif(i<int(args.num_clients*(18/20))): #group3
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group3,100,args.dataset)
                    acc_list3.append(acc)
                else:
                    _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group4,100,args.dataset)
                    acc_list4.append(acc)
                    
    elif(args.iid == 'label2'):
        test_loader = create_test_data_label(args.dataset)
        for i in range(args.num_clients):
            group = np.unique(list(itertools.product(range(10),repeat=2))[i])
            _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group,10,args.dataset)
            acc_list0.append(acc)
            
    elif(args.iid == 'label3'):
        test_loader = create_test_data_label(args.dataset)
        for i in range(args.num_clients):
            print(f"testing {i}")
            if(i in np.arange(0,20)):
                group = cluster_labels[0]
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group, 10, args.dataset)
                acc_list0.append(acc)
            elif(i in np.arange(20,40)):
                group = cluster_labels[1]
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group, 10, args.dataset)
                acc_list1.append(acc)
            elif(i in np.arange(40,60)):
                group = cluster_labels[2]
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group, 10, args.dataset)
                acc_list2.append(acc)
            elif(i in np.arange(60,80)):
                group = cluster_labels[3]
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group, 10, args.dataset)
                acc_list3.append(acc)
            elif(i in np.arange(80,100)):
                group = cluster_labels[4]
                _, acc = test_labelshift(clients[i].best_model, criterion, test_loader, device, group, 10, args.dataset)
                acc_list4.append(acc)
    

    test_acc_0 = np.mean(acc_list0)
    test_acc_1 = np.mean(acc_list1)
    test_acc_2 = np.mean(acc_list2)
    test_acc_3 = np.mean(acc_list3)
    test_acc_4 = np.mean(acc_list4)
    
    with open(experiment_dir+'/'+filename,'a') as f1:
        f1.write(f'{args.n_rounds};{args.n_rounds_pens};{args.num_clients};{args.local_ep};{args.bs};{args.lr};{args.n_clusters};{args.pens};{args.DAC};{args.DAC_var};{args.oracle};{args.random};{args.bandit};{args.bandit_var};{args.top_m};{args.n_sampled};{args.n_data_train};{args.n_data_val};{args.tau};{args.q};{args.q_min};{args.iid};{test_acc_0};{test_acc_1};{test_acc_2};{test_acc_3};{test_acc_4};{args.dataset}')
        f1.write("\n")
