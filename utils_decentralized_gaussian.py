
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:44:40 2020

@author: Lenovo Thinkpad
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:43:57 2020

@author: Lenovo Thinkpad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:58:49 2020

@author: Lenovo Thinkpad
"""
import numpy as np
import math
import pickle
import os
from copy import deepcopy
from scipy.stats import norm


def is_doubly_stochastic(W):
    eps = 10e-2
    m, n = W.shape
    for i in range(m):
        sum1 = 0
        sum2 = 0
        for j in range(n):
            sum1 += W[i][j]
            sum2 += W[j][i]
        if (np.abs(sum1 -1) > eps) or (np.abs(sum2 -1) > eps):
            return False
    return True


def compute_W(N, type, args=None):
    if type == 'cycle':
        A = np.zeros((N, N), dtype=float)
        D_sqrt_inv = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(1,2):
                A[i][(i+j)%N] = 1
                A[i][(i-j)%N] = 1
            A[i][i] = 0
        delta = [A[0][j] for j in range(N)].count(1)
        for i in range(N):
            D_sqrt_inv[i][i] = 1/np.sqrt(delta)
        Lap = np.eye(N, dtype=float) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
        W = np.eye(N, dtype=float) - delta/(delta+1)*Lap
        #if not is_doubly_stochastic(W):
        #    raise RuntimeError("W is not double stochastic")
    elif type == 'complete':
        W = np.ones((N,N), dtype=float)/N
    elif type == '5regular':
        A = np.zeros((N, N), dtype=float)
        D_sqrt_inv = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(1,6):
                A[i][(i+j)%N] = 1
                A[i][(i-j)%N] = 1
            A[i][i] = 1
        delta = [A[0][j] for j in range(N)].count(1)
        for i in range(N):
            D_sqrt_inv[i][i] = 1/np.sqrt(delta)
        Lap = np.eye(N, dtype=float) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
        W = np.eye(N, dtype=float) - delta/(delta+1)*Lap
    elif type == 'grid':
        aux = int(round(math.sqrt(N)))
        if N != aux * aux:
            raise RuntimeError("With type {} the number of nodes n must be a square number. It was {}".format(type, N))
        A = np.zeros((N, N), dtype=float)
        for x in range(N):
            i = x//aux
            j = x % aux
            if (i+1) < aux:
                A[x][aux*(i+1)+j] = 1
            if (i-1) >=0:
                A[x][aux*(i-1)+j] = 1
            if (j+1) < aux:
                A[x][aux*i + j+1] = 1
            if (j-1) >=0:
                A[x][aux*(i) + j-1] = 1
        delta = np.array([[A[i][j] for j in range(N)].count(1) for i in range(N)])
        D_sqrt_inv = np.diag(1/np.sqrt(delta))

        Lap = np.eye(N, dtype=float) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
        # Note that since the grid is not a regular graph, we have to use a different formula for P here. See Duchi et al 2012
        W = np.eye(N, dtype=float) - 1/(delta.max()+1)*np.dot(np.dot(D_sqrt_inv, Lap), D_sqrt_inv)
        #if not is_doubly_stochastic(W):
        #    raise RuntimeError("W is not double stochastic")
    
    if not is_doubly_stochastic(W):
        raise RuntimeError("W is not double stochastic")
    return W

class UCB:
    
    def __init__(self, alpha, theta, sigma):
        
        
        self.alpha = alpha # alpha: exploration parameter for UCB
        self.theta = theta # theta (mean of each Gaussian arm)
        self.sigma = sigma 
        
        self.num_arms = len(self.theta) # Set number of arms
        self.ucb_values = np.zeros((self.num_arms, 1)) # Confidence values for each arm
        # Empirical mean reward for each arm
        self.mu = np.array([np.random.normal(self.theta[arm], sigma) for arm in range(self.num_arms)]) 
        
        self.mu = self.mu.astype(np.float32)
        self.num_arm_pulls = np.ones((self.num_arms, 1)) # Num of plays for each arm
        self.num_plays = self.num_arms # Total number of plays
        self.curr_reward = None
        self.curr_arm = None

    # Return the action
    def select_action(self):
        self.ucb_values = np.array([self.mu[arm] + np.sqrt((self.alpha*np.log(self.num_plays))/self.num_arm_pulls[arm]) 
                                    for arm in range(self.num_arms)]) # construct UCB values
        self.curr_arm = np.argmax(self.ucb_values) # select an optimal arm
        
    # Get reward for the choosen action
    def get_reward_regret(self):
        self.curr_reward = np.random.normal(self.theta[self.curr_arm], self.sigma)
        regret = self.theta.max() - self.curr_reward
        return regret
    
    # Update the model parameters before the next play
    def update(self):
        self.num_plays += 1 # update the total number of plays
        self.num_arm_pulls[self.curr_arm] += 1 # update the step of individual arms
        self.mu[self.curr_arm] = self.mu[self.curr_arm]*(1-(1 / float(self.num_arm_pulls[self.curr_arm]))) + self.curr_reward/float(self.num_arm_pulls[self.curr_arm])
    
    # Play one round
    def play(self):   
        self.select_action() # select action
        regret = self.get_reward_regret() # observe reward for the selected action
        self.update() # update the model
        return regret
    
class Bayes_UCB:
    
    def __init__(self, rho, c, theta, sigma, mean_vec, sigma_vec, T):
        
        self.rho = rho
        self.theta = deepcopy(theta) # theta (mean of each Gaussian arm)
        self.num_arms = len(self.theta) # Set number of arms
        self.sigma = sigma
        
        self.mean_vec = deepcopy(mean_vec) # mean values for posterior Gaussian distributions
        self.sigma_vec = deepcopy(sigma_vec) # variance values for posterior Gaussian distributions 
        self.num_plays = 1 # Total number of plays
        self.T = T # Horizon T
        self.c = c # Constant c
        self.curr_arm = None
        self.curr_reward = None

    # Return the action
    def select_action(self):
        prob = 1- 1/float(self.num_plays*(np.power(np.log(self.T),self.c)))
        self.quantile_values = np.array([norm.ppf(prob, self.mean_vec[arm], self.sigma_vec[arm]) for arm in range(self.num_arms)]) # Quantile values for each arm
        self.curr_arm = np.argmax(self.quantile_values) # select an arm
        
    # Get reward and regret for the choosen action
    def get_reward_regret(self):
        self.curr_reward = np.random.normal(self.theta[self.curr_arm], self.sigma)
        regret = self.theta.max() - self.curr_reward
        return regret
    
    # Update the model parameters before the next play
    def update(self):
        self.num_plays += 1 # update the total number of plays
        var = self.sigma_vec[self.curr_arm]**2
        self.sigma_vec[self.curr_arm] = np.sqrt(1/((1/var)+(self.rho/self.sigma**2))) # update sigma value for the arm
        self.mean_vec[self.curr_arm] = (self.sigma_vec[self.curr_arm]**2)*((self.mean_vec[self.curr_arm]/var)+(self.rho*self.curr_reward/self.sigma**2)) # update mean value for the arm
    
    # Play one round
    def play(self):   
        self.select_action() # select action
        regret = self.get_reward_regret() # observe reward for the selected action
        self.update() # update the model
        return regret
    
class Thompson_sampling:
    
    def __init__(self, rho, max_iter, theta, sigma, mean_vec, sigma_vec):
        
        self.rho = rho
        self.T = max_iter
        self.theta = deepcopy(theta) # theta (mean of each Gaussian arm)
        self.num_arms = len(self.theta) # Set number of arms
        self.sigma = sigma
        
        self.mean_vec = deepcopy(mean_vec) # mean values for posterior Gaussian distributions
        self.sigma_vec = deepcopy(sigma_vec) # variance values for posterior Gaussian distributions 
        self.mu = np.zeros((self.num_arms, 1))
        
        self.curr_reward = None
        self.curr_arm = None
        self.arm_samples = np.random.normal(0, 1, (self.num_arms, self.T+1))
        self.num_plays = 1 # Total number of plays

    # Return the action
    def select_action(self):
        #self.mu = np.multiply(self.arm_samples[:,self.num_plays], self.sigma_vec) + self.mean_vec
        #np.array([self.arm_samples[arm][self.num_plays]*self.sigma_vec[arm]+self.mean_vec[arm] for arm in range(self.num_arms)])
        self.mu = np.array([np.random.normal(self.mean_vec[arm], self.sigma_vec[arm]) for arm in range(self.num_arms)])
        self.curr_arm = np.argmax(self.mu)
        
    # Get reward and regret for the choosen action
    def get_reward_regret(self):
        self.curr_reward = np.random.normal(self.theta[self.curr_arm], self.sigma)
        regret = self.theta.max() - self.curr_reward
        return regret
    
    # Update the model parameters before the next play
    def update(self):
        self.num_plays += 1 # update the total number of plays
        var = self.sigma_vec[self.curr_arm]**2
        self.sigma_vec[self.curr_arm] = np.sqrt(1/((1/var)+(self.rho/self.sigma**2))) # update sigma value for the arm
        self.mean_vec[self.curr_arm] = (self.sigma_vec[self.curr_arm]**2)*((self.mean_vec[self.curr_arm]/var)+(self.rho*self.curr_reward/self.sigma**2)) # update mean value for the arm
    
    # Play one round
    def play(self):   
        self.select_action() # select action
        regret = self.get_reward_regret() # observe reward for the selected action
        self.update() # update the model
        return regret

#### Other utility functions
        
def read_regret(filename):
    fileObject = open(filename,'rb')
    return pickle.load(fileObject)

def run_singleAgent_UCB(max_iter, runs, alpha, theta, sigma, exp_num, save = True, rerun=False):
    
    filename = './data/UCB_N1_T{}_alpha{}_Run{}_GaussianExp{}'.format(max_iter, alpha, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running UCB...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))
             
        for run in range(runs):
            if(run%20 == 0):
                print('Execution number: {}'.format(run))
            mab_ucb = UCB(alpha, theta, sigma) # reinitialize the strategy
            regret = [0]
            for t in range(max_iter):
                curr_regret = mab_ucb.play()
                regret.append(regret[-1] + curr_regret)
            avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(run+1)) + np.multiply(regret, 1/(run+1))
        print('Completed')    
        if save:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fileObject = open(filename,'wb')
            pickle.dump(avg_cumm_regret,fileObject)
            fileObject.close()
            print('Saved')
        
    return avg_cumm_regret

def run_singleAgent_BayesUCB(max_iter, runs, c, mean_vec, sigma_vec, theta, sigma, exp_num, save = True, rerun=False):
    
    rho = 1
    filename = './data/BayesUCB_N1_T{}_Run{}_GaussianExp{}'.format(max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Bayes UCB...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))
     
        for run in range(runs):
            if(run%20 == 0):
                print('Execution number: {}'.format(run))
            mab_bayes_ucb = Bayes_UCB(rho, c, theta, sigma, mean_vec, sigma_vec, max_iter) # reinitialize the strategy
            regret = [0]
            for t in range(max_iter):
                curr_regret = mab_bayes_ucb.play()
                regret.append(regret[-1] + curr_regret)
            avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(run+1)) + np.multiply(regret, 1/(run+1))
        print('Completed')    
        if save:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fileObject = open(filename,'wb')
            pickle.dump(avg_cumm_regret,fileObject)
            fileObject.close()
            print('Saved')
        
    return avg_cumm_regret

def run_singleAgent_TS(max_iter, runs, mean_vec, sigma_vec, theta, sigma, exp_num, save = True, rerun=False):
    
    rho = 1
    filename = './data/TS_N1_T{}_Run{}_GaussianExp{}'.format(max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Thompson Sampling...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))
     
        for run in range(runs):
            if(run%200 == 0):
                print('Execution number: {}'.format(run))
            mab_ts = Thompson_sampling(rho, max_iter, theta, sigma, mean_vec, sigma_vec)# reinitialize the strategy
            regret = [0]
            for t in range(max_iter):
                curr_regret = mab_ts.play()
                regret.append(regret[-1] + curr_regret)
            avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(run+1)) + np.multiply(regret, 1/(run+1))
        print('Completed')    
        if save:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fileObject = open(filename,'wb')
            pickle.dump(avg_cumm_regret,fileObject)
            fileObject.close()
            print('Saved')
        
    return avg_cumm_regret

def run_MultiAgent_TS(max_iter, runs, W, mean_vec, sigma_vec, theta, sigma, exp_num, type_W ='cycle', save = True, rerun=False):
    
    N = W.shape[0]
    rho = N
    filename = './data/TS_N{}_{}_T{}_Run{}_GaussianExp{}'.format(N, type_W, max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Multi-agent Thompson Sampling...') 
        
        num_arms = len(theta)
        mean_mat = np.zeros((N, num_arms))
        sigma_mat = np.zeros((N, num_arms))
        
        avg_cumm_regret = np.zeros((max_iter+1))
        
        for run in range(runs):
            
            if(run%100 == 0):
                print('Execution number: {}'.format(run))
            
            # reinitialize the strategy
            agent_list = []
            for i in range(N):
                curr_class = Thompson_sampling(rho, max_iter, theta, sigma, mean_vec, sigma_vec)
                agent_list.append(curr_class)
                
            netwk_regret = [0]
            for t in range(max_iter):
                '''
                if(t%500 == 0):
                    print('Iteration number is {}'.format(t))
                '''
                curr_netwk_regret = 0
                for i in range(N):
                    curr_regret = agent_list[i].play()
                    curr_netwk_regret += curr_regret
                    
                    # update the alpha and beta matrix to be used later for merging
                    mean_mat[i,:] = deepcopy(agent_list[i].mean_vec)
                    sigma_mat[i,:] = deepcopy(agent_list[i].sigma_vec)
                
                # Merge marginal posteriors at each agent
                mean_mat = np.array(mean_mat)
                sigma_mat = np.array(sigma_mat)
                
                inv_sigma_sq = np.reciprocal(np.square(sigma_mat))
                merged_sigma = np.reciprocal(np.sqrt(np.matmul(W, inv_sigma_sq)))
                inv_sigma_mu = np.multiply(inv_sigma_sq, mean_mat)
                merged_mean = np.multiply(np.square(merged_sigma),np.matmul(W, inv_sigma_mu))
                
                for i in range(N):
                    agent_list[i].sigma_vec = merged_sigma[i,:]
                    agent_list[i].mean_vec = merged_mean[i,:]

                    
                netwk_regret.append(netwk_regret[-1] + curr_netwk_regret/N)
                
                
            # avg cumm network-wide regret    
            avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(run+1)) + np.multiply(netwk_regret, 1/(run+1))
            
            
        print('Completed')    
        if save:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fileObject = open(filename,'wb')
            pickle.dump(avg_cumm_regret,fileObject)
            fileObject.close()
            print('Saved')
        
    return avg_cumm_regret    


    

def run_MultiAgent_BayesUCB(max_iter, runs, W, c, mean_vec, sigma_vec, theta, sigma, exp_num, type_W ='cycle', save = True, rerun=False):
    
    N = W.shape[0]
    rho = N
    filename = './data/BayesUCB_N{}_{}_T{}_Run{}_GaussianExp{}'.format(N, type_W, max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Multi-agent Bayes UCB')
        
        
        num_arms = len(theta)
        mean_mat = np.zeros((N, num_arms))
        sigma_mat = np.zeros((N, num_arms))
        
        avg_cumm_regret = np.zeros((max_iter+1))
        
        for run in range(runs):
            
            if(run%100 == 0):
                print('Execution number: {}'.format(run))
            
            # reinitialize the strategy
            agent_list = []
            for i in range(N):
                curr_class = Bayes_UCB(rho, c, theta, sigma, mean_vec, sigma_vec, max_iter) 
                agent_list.append(curr_class)
                
            netwk_regret = [0]
            for t in range(max_iter):
                
                '''
                if t%500 == 0:
                    print('Iteration number is {}'.format(t))
                '''

                curr_netwk_regret = 0
                for i in range(N):
                    curr_regret = agent_list[i].play()
                    curr_netwk_regret += curr_regret
                    
                    # update the alpha and beta matrix to be used later for merging
                    mean_mat[i,:] = deepcopy(agent_list[i].mean_vec)
                    sigma_mat[i,:] = deepcopy(agent_list[i].sigma_vec)
    

                # Merge marginal posteriors at each agent
                mean_mat = np.array(mean_mat)
                sigma_mat = np.array(sigma_mat)
                
                inv_sigma_sq = np.reciprocal(np.square(sigma_mat))
                merged_sigma = np.reciprocal(np.sqrt(np.matmul(W, inv_sigma_sq)))
                inv_sigma_mu = np.multiply(inv_sigma_sq, mean_mat)
                merged_mean = np.multiply(np.square(merged_sigma),np.matmul(W, inv_sigma_mu))

                for i in range(N):
                    agent_list[i].sigma_vec = merged_sigma[i,:]
                    agent_list[i].mean_vec = merged_mean[i,:]
                '''
                for i in range(N):
                    agent_list[i].merge_posteriors(W[i,:], alpha_mat, beta_mat)
                '''
                
                netwk_regret.append(netwk_regret[-1] + curr_netwk_regret/N)
                
                
            # avg cumm network-wide regret    
            avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(run+1)) + np.multiply(netwk_regret, 1/(run+1))
            
            
        print('Completed')    
        if save:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fileObject = open(filename,'wb')
            pickle.dump(avg_cumm_regret,fileObject)
            fileObject.close()
            print('Saved')
        
    return avg_cumm_regret