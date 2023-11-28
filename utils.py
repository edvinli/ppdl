import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from torchvision import transforms
import itertools
import math
from itertools import combinations
import operator
import scipy.optimize as opt
from loguru import logger
import abc

from models import *

class CMAB_Player:
    def __init__(self, num_clients, group_size, q, indx=None):
        # all different way to choose clients in groups
        self.numArms = math.comb(num_clients - 1, group_size)
        self.num_actions = self.numArms

        self.group_size = group_size

        self.pulls = np.zeros(self.numArms)
        self.empReward = np.zeros(self.numArms)
        self.empReward[:] = np.inf

        self.sumReward = np.zeros(self.numArms)
        self.Index = dict(zip(range(self.numArms), [np.inf] * self.numArms))
        self.indx = indx
        # Create mapping between arm and members in the group
        clients = set(range(num_clients)) - set([indx])
        groups = combinations(clients, group_size)

        # we need a binary representation for what clients are inside each group
        self.group_bin = np.zeros((self.numArms, num_clients), dtype=bool)
        for i, group in enumerate(groups):  # loop over all possible groups
            self.group_bin[i, group] = True

        # empPseudoReward will be built over time
        self.empPseudoReward = np.zeros((self.numArms, 0), dtype=np.float16)
        self.action_to_pseudoreward = np.zeros(0)

        # This is used to form the pseudo rewards
        self.q = q #0.15

    def _update_competitive_set(self, time):
        # Step 1: identify the set S_t of significant arms
        # add to set \ell for arms with pulls >t/K
        bool_ell = self.pulls >= (float(time - 1) / self.numArms)

        max_mu_hat = np.max(
            self.empReward[bool_ell]
        )  # k^emp(t), this will be used as the reference to define competitive arms

        if self.empReward[bool_ell].shape[0] == 1:
            secmax_mu_hat = max_mu_hat
        else:
            temp = self.empReward[bool_ell]
            temp[::-1].sort()
            secmax_mu_hat = temp[1]
        argmax_mu_hat = np.where(self.empReward == max_mu_hat)[0][
            0
        ]  # arm with highest reward among significant arms

        # Step 2: Identify set of competitive arms
        # Set of competitive arms - update through the run
        if self.empPseudoReward.shape[1] == 0:
            min_phi = np.ones((self.num_actions, 1)) * np.inf
        else:
            pulled_arms = np.where(bool_ell == True)[0]  # get index of significant arms
            indx = np.where(np.in1d(self.action_to_pseudoreward, pulled_arms))[
                0
            ]  # map these arms to where they are stored in pseudo rewards
            min_phi = np.min(
                self.empPseudoReward[:, indx], axis=1
            )  # get the smallest empirical pseudo reward for each arm, this corresponds to the tightest upper bound on each arm based on correlations

        comp_set = set()
        # Adding back the arm with largest empirical mean
        comp_set.add(argmax_mu_hat)

        for arm in range(self.numArms):
            if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                comp_set.add(arm)
            elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                comp_set.add(arm)

        return comp_set

    def get_pseudo_rewards(self, k_t, reward):
        """This function is important as it decides how much one gets to know about other arms.
        With a larger q, less is known.

        Args:
            k_t (_type_): pulled arm
            reward (_type_): reward scalar from pulling the arm

        Returns:
            _type_: _description_
        """

        chosen_group = self.group_bin[k_t]
        pseudo_rewards = np.ones(self.numArms)  # all rewards are in [0, 1]
        overlaps = np.sum(np.bitwise_and(chosen_group, self.group_bin), axis=1)

        # find all groups that overlap and update their pseudo regret
        for i, val in enumerate(overlaps):
            if (val < self.group_size) and (val > 0):
                pseudo_rewards[i] = np.minimum(reward + self.q / val, 1.0)
            elif val == self.group_size:  # all overlap
                pseudo_rewards[i] = reward

        return pseudo_rewards

    @abc.abstractmethod
    def sample_action(self):
        """Abstract method that is game specific"""
        return

    @abc.abstractmethod
    def update_policy(self):
        """Abstract method to evaluate the upper bound on the pseudo regret"""
        return
    
class MAB_Player:
    """A class used to represent a leader-follower game"""
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.cumulative_losses = np.zeros(self.num_actions) # Keep track of the losses observed for each action
        self.last_played_action = -1 # keep track of latest action
        self.weights = np.full((self.num_actions), 1/self.num_actions) # current strategy
            
    @abc.abstractmethod
    def sample_action(self):
        """Abstract method that is game specific"""
        return

    @abc.abstractmethod
    def update_policy(self):
        """Abstract method to evaluate the upper bound on the pseudo regret"""
        return

    def reset(self):
        """Reset the strategy to the uniform strategy"""
        self.weights = np.full((self.num_actions), 1/self.num_actions);       
        self.cumulative_losses = np.zeros(self.num_actions) # Keep track of the losses observed for each action
        self.last_played_action = -1 # keep track of latest action
        
    def get_policy(self):
        """Read out the policy"""
        return self.weights
    
"""Imlementation of the Tsallis-INF algorithm with alpha=0.5"""
class TsallisInf(MAB_Player):
    def __init__(self, num_actions):
        super().__init__(num_actions)
        self.alpha = 0.5 # alpha value in the Tsallis-Inf algorithm

    def sample_action(self):
        """Sample an action based on the strategy."""
        self.last_played_action = np.random.choice(a=self.num_actions, p=self.weights)
        return self.last_played_action

    def update_policy(self, reward, time):
        """Update the strategy based on the observed reward using the Tsallis-Inf algorithm"""
        # for a reward in [0,1], loss = 1 - reward
        biased_loss = 1.0 - reward
        # unbiased estimate, from the weights of the previous step
        unbiased_loss = biased_loss / self.weights[self.last_played_action]
        self.cumulative_losses[self.last_played_action] += unbiased_loss
        eta_t = 1.0 / np.sqrt(max(1,time))
        
        # solve f(x)=1 to get an approximation of the (unique) Lagrange multiplier x
        def objective_function(x):
            return (np.sum( (eta_t * (self.cumulative_losses - x + np.finfo(float).eps)) ** -2) - 1)**2 
        result_of_minimization = opt.minimize_scalar(objective_function)
        x = result_of_minimization.x
        
        #  use x to compute the new weights
        new_weights =  ( eta_t * (self.cumulative_losses - x) ) ** -2
        
        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with new_weights=[1/K...]
        if not np.all(np.isfinite(new_weights)):
            new_weights[:] = 1.0
        # 3. Renormalize weights at each step
        new_weights /= np.sum(new_weights)
        # 4. store weights
        self.weights =  new_weights
        
class CTsallisInf(CMAB_Player):
    """Imlementation of the Tsallis-INF algorithm with alpha=0.5"""

    def __init__(self, num_clients, group_size, q, indx=None):
        super().__init__(num_clients, group_size, q, indx)
        self.alpha = 0.5  # alpha value in the Tsallis-Inf algorithm
        self.comp_set = np.arange(0, self.num_actions)
        self.weights = np.full(
            (self.num_actions), 1 / self.num_actions
        )  # current strategy
        self.cumulative_losses = np.zeros(
            self.num_actions
        )  # Keep track of the losses observed for each action
        self.last_played_action = -1  # keep track of latest action

    def sample_action(self, time):
        """Sample an action based on the strategy."""
        self.comp_set = np.array(list(self._update_competitive_set(time)))
        #logger.info(f"Size of competitive set: {len(self.comp_set)}")
        num_comp_arms = self.comp_set.shape[0]

        comp_weights = self.weights[self.comp_set] + 1e-10
        comp_weights /= sum(comp_weights)
        self.action_index = np.random.choice(a=num_comp_arms, p=comp_weights)
        self.last_played_action = self.comp_set[self.action_index]
        logger.info(f"Playing action: {self.last_played_action}")
        self.pulls[self.last_played_action] = self.pulls[self.last_played_action] + 1
        return self.last_played_action

    def compute_new_weights(self, reward, time):
        eta_t = 1.0 / np.sqrt(max(1, time))

        biased_loss = 1.0 - reward
        # unbiased estimate, from the weights of the previous step
        unbiased_loss = biased_loss / (self.weights[self.last_played_action] + 1e-10)
        self.cumulative_losses[self.last_played_action] += unbiased_loss

        # Evaluate weights for competitive arms
        # solve f(x)=1 to get an approximation of the (unique) Lagrange multiplier x
        def objective_function(x):
            return (
                np.sum(
                    (eta_t * (self.cumulative_losses - x + np.finfo(float).eps)) ** -2
                )
                - 1
            ) ** 2

        result_of_minimization = opt.minimize_scalar(objective_function)
        x = result_of_minimization.x

        #  use x to compute the new weights
        new_weights = (eta_t * (self.cumulative_losses - x)) ** -2

        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with new_weights=[1/K...]
        if not np.all(np.isfinite(new_weights)):
            new_weights[:] = 1.0
        # Renormalize weights
        new_weights /= np.sum(new_weights)
        # store weights
        self.weights = new_weights

    def update_policy(self, k_t, reward, time):
        """Update the policy based on the pulled arm and reward

        Args:
            k_t (_type_): _description_
            reward (_type_): _description_
            time (_type_): _description_
        """
        numArms = self.numArms
        B = [1.0] * numArms  # all rewards are in [0, B]

        # Update \mu_{k_t}
        self.sumReward[k_t] = self.sumReward[k_t] + reward  # sum of rewards for arm
        self.empReward[k_t] = self.sumReward[k_t] / float(
            self.pulls[k_t]
        )  # empirical rewards for arm

        # get the pseudo-rewards of the overlapping groups
        pseudoRewards = self.get_pseudo_rewards(k_t, reward)
        pseudoRewards = np.expand_dims(pseudoRewards, 1)

        if self.pulls[k_t] > 1:
            indx = np.where(self.action_to_pseudoreward == k_t)[0]
            scale = float(self.pulls[k_t] - 1) / float(self.pulls[k_t])
            self.empPseudoReward[:, indx] = scale * self.empPseudoReward[
                :, indx
            ] + np.divide(pseudoRewards, float(self.pulls[k_t]))
            self.empPseudoReward[k_t, indx] = self.empReward[k_t]
        else:
            self.action_to_pseudoreward = np.hstack([self.action_to_pseudoreward, k_t])
            self.empPseudoReward = np.hstack([self.empPseudoReward, pseudoRewards])
            self.empPseudoReward[k_t, -1] = self.empReward[k_t]

        # Diagonal elements of pseudorewards

        self.compute_new_weights(reward, time)
        
class CUCB(CMAB_Player):
    """Imlementation of the C-UCB algorithm"""

    def __init__(self, num_clients, group_size):
        super().__init__(num_clients, group_size)

    def sample_action(self, time):
        """Sample an action based on the strategy."""
        comp_set = self._update_competitive_set(time)
        logger.info(f"Size of competitive set: {len(comp_set)}")
        # Step 3: Play bandit algorithm from competitive arms
        # pull an arm
        if len(comp_set) == 0:
            # UCB for empty comp set
            k_t = max(self.Index.items(), key=operator.itemgetter(1))[0]
        else:
            comp_Index = {ind: self.Index[ind] for ind in comp_set}
            k_t = max(comp_Index.items(), key=operator.itemgetter(1))[0]

        self.pulls[k_t] = self.pulls[k_t] + 1
        logger.info(f"Playing action: {k_t}")
        return k_t

    def update_policy(self, k_t, reward, time):
        """Update the policy based on the pulled arm and reward

        Args:
            k_t (_type_): _description_
            reward (_type_): _description_
            time (_type_): _description_
        """
        numArms = self.numArms
        B = [1.0] * numArms  # all rewards are in [0, B]

        # Update \mu_{k_t}
        self.sumReward[k_t] = self.sumReward[k_t] + reward  # sum of rewards for arm
        self.empReward[k_t] = self.sumReward[k_t] / float(
            self.pulls[k_t]
        )  # empirical rewards for arm

        # get the pseudo-rewards of the overlapping groups
        pseudoRewards = self.get_pseudo_rewards(k_t, reward)
        pseudoRewards = np.expand_dims(pseudoRewards, 1)

        if self.pulls[k_t] > 1:
            indx = np.where(self.action_to_pseudoreward == k_t)[0]
            scale = float(self.pulls[k_t] - 1) / float(self.pulls[k_t])
            self.empPseudoReward[:, indx] = scale * self.empPseudoReward[
                :, indx
            ] + np.divide(pseudoRewards, float(self.pulls[k_t]))
            self.empPseudoReward[k_t, indx] = self.empReward[k_t]
        else:
            self.action_to_pseudoreward = np.hstack([self.action_to_pseudoreward, k_t])
            self.empPseudoReward = np.hstack([self.empPseudoReward, pseudoRewards])
            self.empPseudoReward[k_t, -1] = self.empReward[k_t]

        # Update UCB+LCB indices: Using empirical rewards
        for k in range(numArms):
            if self.pulls[k] > 0:
                # UCB index
                self.Index[k] = self.empReward[k] + B[k] * np.sqrt(
                    2.0 * np.log(time + 1) / self.pulls[k]
                )
                
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, transform):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform:
            image = self.transform(image)
        return image, label


class ClientUpdate(object):
    def __init__(self, train_set, idxs_train, idxs_val, dataset, criterion, lr, device, batch_size, shift, rot_deg, num_users, num_actions, n_sampled, q, idx):
        self.device = device
        self.criterion = criterion
        self.lr = lr
        if('covariate' in shift):
            rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
        else:
            rot_transform = None
            
        self.train_set = DatasetSplit(train_set, idxs_train, rot_transform)
        self.ldr_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
            
        if(idxs_val):
            if('covariate' in shift):
                rot_transform = transforms.RandomRotation(degrees=(rot_deg,rot_deg))
                
            else:
                rot_transform = None
                
            self.val_set = DatasetSplit(train_set, idxs_val, rot_transform)
            self.ldr_val = DataLoader(self.val_set, batch_size = 1, shuffle=False)
        else:
            self.ldr_val = None
        if(dataset=='cifar10'):
            self.local_model = CNN2(num_classes=10).to(self.device)
            self.best_model = copy.deepcopy(self.local_model)
        elif(dataset=='cifar100'):
            self.local_model = CNN2(num_classes=100).to(self.device)
            self.best_model = copy.deepcopy(self.local_model)
        elif(dataset=='fashion-mnist'):
            self.local_model = CNNFashion(num_classes=10).to(self.device)
            self.best_model = copy.deepcopy(self.local_model)
        
        self.received_models = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.n_received = 0
        self.n_sampled = np.zeros(num_users)
        self.n_selected = np.zeros(num_users)
        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.count = 0
        self.stopped_early = False
        
        self.priors = np.zeros(num_users)
        self.priors_norm = np.zeros(num_users)

        self.similarity_scores = np.zeros(num_users)
        self.neighbour_list = []
        
        self.rewards = []
        #self.player = TsallisInf(num_actions)
        #self.player2 = CUCB(num_users,n_sampled)
        self.player3 = CTsallisInf(num_users, n_sampled, q, idx)
        
    def train(self,n_epochs):
        self.local_model.train()
        #optimizer = torch.optim.SGD(self.local_model.parameters(),lr=self.lr)
        optimizer = torch.optim.Adam(self.local_model.parameters(),lr=self.lr)
        
        epoch_loss = []
        
        for iter in range(n_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = self.local_model(images.float())
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                train_loss = sum(batch_loss)/len(batch_loss)
            epoch_loss.append(train_loss)
        
        self.train_loss_list.append(epoch_loss[-1])
        if(self.ldr_val):
            val_loss, val_acc = self.validate(self.local_model, train_set = False)
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
        
            if(val_loss < self.best_val_loss):
                self.count = 0
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_model.load_state_dict(self.local_model.state_dict())
            else:
                self.count += 1
            
        
        return self.best_model, epoch_loss[-1], self.best_val_loss, self.best_val_acc
    
    def validate(self,model,train_set):
        if(train_set):
            ldr = self.ldr_train
        else:
            ldr = self.ldr_val
            
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(ldr):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                log_probs = model(inputs)
                _, predicted = torch.max(log_probs.data, 1)
                                         
                loss = self.criterion(log_probs,labels)                
                batch_loss.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            val_loss = sum(batch_loss)/len(batch_loss)

        return val_loss, val_acc
    
def sample_labels_iid(dataset, num_users, n_data_train, n_data_val):
    """
    Sample I.I.D. (labels) client data from MNIST/CIFAR10/FASHION-MNIST datasets
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_val = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, int(n_data_train), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        dict_users_val[i] = set(np.random.choice(all_idxs, int(n_data_val), replace=False))
        all_idxs = list(set(all_idxs) - dict_users_val[i])
        
    return dict_users, dict_users_val


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def sample_cifargroups(dataset, num_users, n_data_train, n_data_val):

    group1 = np.array([0,1,8,9]) #vehicles
    group2 = np.array([2,3,4,5,6,7]) #animals
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    idxs1, idxs2 = np.array([]), np.array([])
    idxs1 = idxs1.astype(int)
    idxs2 = idxs1.astype(int)
    for x in group1:
        idxs1 = np.append(idxs1, idxs[x == labels[idxs]])
    
    for x in group2:
        idxs2 = np.append(idxs2, idxs[x == labels[idxs]])
        
    print(len(idxs1))
    print(len(idxs2))
    
    for i in range(num_users):
        if(i<int(num_users*0.4)): #vehicles
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
            
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
        else: #animals
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
            
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
        
    return dict_users, dict_users_val

# Sample non-iid CIFAR-10 or Fashion-MNIST, each client belongs to one of five clusters, each cluster defined by 2 random labels
def cifar_noniid_5k(dataset, num_users, n_data, n_data_val):
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_classes = len(label_list)
    user_majority_labels = []

    cluster_labels = []
    for k in range(5):
        majority_labels = np.random.choice(np.unique(label_list), 2, replace = False)
        cluster_labels.append(majority_labels)
        label_list = np.array(list(set(label_list) - set(majority_labels)))
        
    for i in range(num_users):
        if(i in np.arange(0,20)):
            majority_labels = cluster_labels[0]
        elif(i in np.arange(20,40)):
            majority_labels = cluster_labels[1]
        elif(i in np.arange(40,60)):
            majority_labels = cluster_labels[2]
        elif(i in np.arange(60,80)):
            majority_labels = cluster_labels[3]
        elif(i in np.arange(80,100)):
            majority_labels = cluster_labels[4]
            
        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)
        
        #train set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(n_data/2), replace = False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(n_data/2), replace = False)
        
        dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
        dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))
        
        #validation set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(n_data_val/2), replace = False)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(n_data_val/2), replace = False)
        
        dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1_val)))
        dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2_val)))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))

    for i in range(num_users):
        print("Train")
        majority_labels = user_majority_labels[i]
        print("client %d %.2f %d " %(i, (sum(labels[dict_users[i]] == majority_labels[0])+sum(labels[dict_users[i]] == majority_labels[0]))/len(dict_users[i]),len(dict_users[i]) ))
        print(majority_labels)
        if i == range(num_users)[-1]:
            print(10*"-")

    return dict_users, dict_users_val, cluster_labels

#Sample non-iid CIFAR-10 or Fashion-MNIST, 2 random labels in each client
def cifar_noniid2(dataset, num_users, p, n_data, n_data_val, overlap):
    idxs = np.arange(len(dataset),dtype=int)
    labels = np.array(dataset.targets)
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_classes = len(label_list)
    user_majority_labels = []
    overlap_list = list(itertools.combinations(range(num_classes), 2))

    for i in range(num_users):
    #Sample majority class for each user

        if(overlap):
            majority_labels = list(itertools.product(range(num_classes),repeat=2))[i]
        else:
            majority_labels = np.random.choice(np.unique(label_list), 2, replace = False)

        label_list = np.array(list(set(label_list) - set(majority_labels)))
        label1 = majority_labels[0]
        label2 = majority_labels[1]
        majority_labels = np.array([label1, label2])
        user_majority_labels.append(majority_labels)
        
        #train set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p*n_data/2), replace = False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p*n_data/2), replace = False)
        
        dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
        dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))
        
        #validation set
        majority_labels1_idxs = idxs[majority_labels[0] == labels[idxs]]
        majority_labels2_idxs = idxs[majority_labels[1] == labels[idxs]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p*n_data_val/2), replace = False)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p*n_data_val/2), replace = False)
        
        dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1_val)))
        dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2_val)))

        idxs = np.array(list(set(idxs) - set(sub_data_idxs1)))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs2)))
        
    if p<1.0:
        for i in range(num_users):
            if(len(idxs)>=n_data):
                majority_labels = user_majority_labels[i]
                #train set
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11 = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data), replace = False)
                dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs11)))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
                
                #validation set
                non_majority_labels1_idxs = idxs[(majority_labels[0] != labels[idxs]) & (majority_labels[1] != labels[idxs])]
                sub_data_idxs11_val = np.random.choice(non_majority_labels1_idxs, int((1-p)*n_data_val), replace = False)
                dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs11_val)))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs11)))
                
            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))

    for i in range(num_users):
        print("Train")
        majority_labels = user_majority_labels[i]
        print("client %d %.2f %d " %(i, (sum(labels[dict_users[i]] == majority_labels[0])+sum(labels[dict_users[i]] == majority_labels[0]))/len(dict_users[i]),len(dict_users[i]) ))
        print(majority_labels)
        if i == range(num_users)[-1]:
            print(10*"-")

    return dict_users, dict_users_val

def sample_cifar100_groups(dataset, num_users, n_data_train, n_data_val):

    group1 = np.array([0,1,7,8,11,12,13,14,15,16]) #animals
    group2 = np.array([2,4,17]) #trees and plants
    group3 = np.array([3,5,6]) #things
    group4 = np.array([9,10]) #nature and manmade buildings
    group5 = np.array([18,19]) #vehicles
    
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}
    
    idxs = np.arange(len(dataset),dtype=int)
    labels = sparse2coarse(np.array(dataset.targets))
    label_list = np.unique(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)
    idxs = idxs_labels[0,:]
    idxs = idxs.astype(int)
    
    idxs1, idxs2, idxs3, idxs4, idxs5 = np.array([]), np.array([]),  np.array([]), np.array([]),  np.array([])
    idxs1 = idxs1.astype(int)
    idxs2 = idxs2.astype(int)
    idxs3 = idxs3.astype(int)
    idxs4 = idxs4.astype(int)
    idxs5 = idxs5.astype(int)
    
    idxs1 = idxs[np.isin(labels[idxs], group1)]
    idxs2 = idxs[np.isin(labels[idxs], group2)]
    idxs3 = idxs[np.isin(labels[idxs], group3)]
    idxs4 = idxs[np.isin(labels[idxs], group4)]
    idxs5 = idxs[np.isin(labels[idxs], group5)]
        
    print(len(idxs1))
    print(len(idxs2))
    print(len(idxs3))
    print(len(idxs4))
    print(len(idxs5))
    
    for i in range(num_users):
        #print(i)
        if(i<int(num_users*(10/20))): #group1
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
            
            sub_data_idxs1 = np.random.choice(idxs1, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs1)))
            idxs1 = np.array(list(set(idxs1) - set(sub_data_idxs1)))
            
        elif(i<int(num_users*(13/20))): #group2
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
            
            sub_data_idxs2 = np.random.choice(idxs2, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs2)))
            idxs2 = np.array(list(set(idxs2) - set(sub_data_idxs2)))
             
        elif(i<int(num_users*(16/20))): #group3
            sub_data_idxs3 = np.random.choice(idxs3, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs3)))
            idxs3 = np.array(list(set(idxs3) - set(sub_data_idxs3)))
            
            sub_data_idxs3 = np.random.choice(idxs3, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs3)))
            idxs3 = np.array(list(set(idxs3) - set(sub_data_idxs3)))
        elif(i<int(num_users*(18/20))): #group4
            sub_data_idxs4 = np.random.choice(idxs4, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs4)))
            idxs4 = np.array(list(set(idxs4) - set(sub_data_idxs4)))
            
            sub_data_idxs4 = np.random.choice(idxs4, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs4)))
            idxs4 = np.array(list(set(idxs4) - set(sub_data_idxs4)))
        else: #group5
            sub_data_idxs5 = np.random.choice(idxs5, int(n_data_train), replace=False)
            dict_users[i] = list(np.concatenate((dict_users[i], sub_data_idxs5)))
            idxs5 = np.array(list(set(idxs5) - set(sub_data_idxs5)))
            
            sub_data_idxs5 = np.random.choice(idxs5, int(n_data_val), replace=False)
            dict_users_val[i] = list(np.concatenate((dict_users_val[i], sub_data_idxs5)))
            idxs5 = np.array(list(set(idxs5) - set(sub_data_idxs5)))

        
    return dict_users, dict_users_val


def sample_iid(dataset, num_users, n_data_train, n_data_val):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    #num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users_val = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, int(n_data_train), replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        dict_users_val[i] = set(np.random.choice(all_idxs, int(n_data_val), replace=False))
        all_idxs = list(set(all_idxs) - dict_users_val[i])
        
    return dict_users, dict_users_val

def test(model,criterion,test_loader,device,num_classes):
    #model.to('cpu')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(model(vec).shape)
            log_probs = model(inputs).view(1,num_classes)

            _, predicted = torch.max(log_probs.data, 1)
            test_loss += criterion(log_probs, labels.long()).item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total
    return test_loss, test_acc


def test_labelshift(model,criterion,test_loader,device,group_labels,num_classes,dataset):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            if(dataset=='cifar100'):
                labels = sparse2coarse(labels)
            if(labels.item() in group_labels):
                log_probs = model(inputs).view(1,num_classes)

                _, predicted = torch.max(log_probs.data, 1)
                test_loss += criterion(log_probs, labels.long()).item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
    test_loss /= len(test_loader.dataset)
    test_acc = 100 * correct / total
    return test_loss, test_acc
