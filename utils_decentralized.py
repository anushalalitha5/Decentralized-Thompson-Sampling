# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:58:49 2020

@author: Lenovo Thinkpad
"""
import numpy as np
import random
import math
import pickle
import os
from copy import deepcopy
from scipy.stats import beta


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

def compute_W_gossip(N, type, max_iter, args=None):
    W_seq = np.zeros((max_iter, N, N), dtype=float)
    I = np.eye(N, dtype=float)
    if type == 'complete':
        for t in range(max_iter):
            i = random.randint(1,N)
            j = random.randint(1,N)
            while i == j:
                i = random.randint(1,N)
                j = random.randint(1,N)
            V = (I[:,i]-I[:,j]).reshape(N,1)
            W_seq[t, :] = I-0.5*np.dot(V, V.T)
    else: 
        raise RuntimeError("Not implemented")
    return W_seq

def compute_W_linkfailure(N, type, p, max_iter, args=None):
    W_seq = np.zeros((max_iter, N, N), dtype=float)
    if type == 'complete':
        for t in range(max_iter):
            A = np.zeros((N, N), dtype=float)
            D_sqrt_inv = np.zeros((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    A[i][j] = np.random.binomial(1,p)
            delta = [A[0][j] for j in range(N)].count(1)
            for i in range(N):
                D_sqrt_inv[i][i] = 1/np.sqrt(delta)
                Lap = np.eye(N, dtype=float) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
            W_seq[t,:] = np.eye(N, dtype=float) - delta/(delta+1)*Lap
    else: 
        raise RuntimeError("Not implemented")
    return W_seq

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
            for j in range(1,3):
                A[i][(i+j)%N] = 1
                A[i][(i-j)%N] = 1
            A[i][i] = 0
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

### Classes for single agent

class UCB:
    
    def __init__(self, alpha, theta):
        
        self.alpha = alpha # alpha: exploration parameter for UCB
        self.theta = theta # theta (mean of each Bernoulli arm)
        self.num_arms = len(self.theta) # Set number of arms
        self.ucb_values = np.zeros((self.num_arms, 1)) # Confidence values for each arm
        self.mu = np.array([np.random.binomial(1,self.theta[arm]) for arm in range(self.num_arms)]) # Empirical mean reward for each arm
        
        self.mu = self.mu.astype(np.float32)
        self.num_arm_pulls = np.ones((self.num_arms, 1)) # Num of plays for each arm
        self.num_plays = self.num_arms # Total number of plays
        self.curr_arm = None 
        self.curr_reward = None

    # Return the action
    def select_action(self):
        self.ucb_values = np.array([self.mu[arm] + np.sqrt((self.alpha*np.log(self.num_plays))/(2*self.num_arm_pulls[arm])) for arm in range(self.num_arms)]) # construct UCB values
        self.curr_arm = np.argmax(self.ucb_values) # select an optimal arm
        
    # Get reward for the choosen action
    def get_reward_regret(self):
        self.curr_reward = np.random.binomial(1,self.theta[self.curr_arm])
        regret = self.theta.max() - self.curr_reward
        return regret
    
    # Update the model parameters before the next play
    def update(self):
        self.num_plays += 1 # update the total number of plays
        self.num_arm_pulls[self.curr_arm] += 1 # update the step of individual arms
        self.mu[self.curr_arm] = self.mu[self.curr_arm] + (self.curr_reward- self.mu[self.curr_arm])/float(self.num_arm_pulls[self.curr_arm])
        return
    
    # Play one round
    def play(self):   
        self.select_action() # select action
        regret = self.get_reward_regret() # observe reward for the selected action
        self.update() # update the model
        return regret

class Bayes_UCB:
    
    def __init__(self, rho, c, theta, alpha_vec, beta_vec, T):
        
        self.theta = deepcopy(theta) # theta (mean of each Bernoulli arm)
        self.num_arms = len(self.theta) # Set number of arms
        self.alpha_vec = deepcopy(alpha_vec) # alpha values for positerior beta distribution
        self.beta_vec = deepcopy(beta_vec) # beta values for posterior beta distribution 
        self.num_plays = 1 # Total number of plays
        self.T = T # Horizon T
        self.c = c # Constant c
        self.curr_arm = None 
        self.curr_reward = None
        self.rho = rho

    # Return the action
    def select_action(self):
        prob = 1- 1/float(self.num_plays*(np.power(np.log(self.T),self.c)))
        self.quantile_values = np.array([beta.ppf(prob, self.alpha_vec[arm], self.beta_vec[arm]) for arm in range(self.num_arms)]) # Quantile values for each arm
        self.curr_arm = np.argmax(self.quantile_values) # select an arm
        
    # Get reward and regret for the choosen action
    def get_reward_regret(self):
        self.curr_reward = np.random.binomial(1,self.theta[self.curr_arm])
        regret = self.theta.max() - self.curr_reward
        return regret
    
    # Update the model parameters before the next play
    def update(self):
        self.num_plays += 1 # update the total number of plays
        self.alpha_vec[self.curr_arm] += self.rho*self.curr_reward # update alpha value for the arm
        self.beta_vec[self.curr_arm] += self.rho*(1 - self.curr_reward) # update alpha value for the arm
        return
    
    # Play one round
    def play(self):   
        self.select_action() # select action
        regret = self.get_reward_regret() # observe reward for the selected action
        self.update() # update the model
        return regret
    
    # Merge marginal posteriors
    def merge_posteriors(self, W_vec, alpha_mat, beta_mat):
        # Merge posterior for each arm
        self.alpha_vec = np.array([np.dot(W_vec, alpha_mat[arm,:]) for arm in range(self.num_arms)])
        self.beta_vec = np.array([np.dot(W_vec, beta_mat[arm,:]) for arm in range(self.num_arms)])
        

class Thompson_sampling:
    
    def __init__(self, rho, theta, alpha_vec, beta_vec):
        
        self.rho = rho
        self.theta = deepcopy(theta) # theta (mean of each Bernoulli arm)
        self.num_arms = len(self.theta) # Set number of arms
        self.alpha_vec = deepcopy(alpha_vec) # alpha values for beta distribution
        self.beta_vec = deepcopy(beta_vec) # beta values for beta distribution
        self.mu = np.zeros((self.num_arms, 1))
        self.curr_arm = None
        self.curr_reward = None

    # Return the action
    def select_action(self):
        self.mu = np.array([beta.rvs(self.alpha_vec[arm], self.beta_vec[arm]) for arm in range(self.num_arms)])
        self. curr_arm = np.argmax(self.mu)
        
    # Get reward and regret for the choosen action
    def get_reward_regret(self):
        self.curr_reward = np.random.binomial(1,self.theta[self.curr_arm])
        regret = self.theta.max() - self.curr_reward
        return regret
    
    # Update the model parameters before the next play
    def update(self):
        self.alpha_vec[self.curr_arm] += self.rho*self.curr_reward # update alpha value for the arm
        self.beta_vec[self.curr_arm] += self.rho*(1 - self.curr_reward) # update alpha value for the arm
    
    # Play one round
    def play(self):   
        self.select_action() # select action
        regret = self.get_reward_regret() # observe reward for the selected action
        self.update() # update the model
        return regret
    
    # Merge marginal posteriors
    def merge_posteriors(self, W_vec, alpha_mat, beta_mat):
        # Merge posterior for each arm
        self.alpha_vec = np.array([np.dot(W_vec, alpha_mat[arm,:]) for arm in range(self.num_arms)])
        self.beta_vec = np.array([np.dot(W_vec, beta_mat[arm,:]) for arm in range(self.num_arms)])
        
        
### Other utility functions
        
def read_regret(filename):
    fileObject = open(filename,'rb')
    return pickle.load(fileObject)

def run_singleAgent_UCB(max_iter, runs, alpha, theta, exp_num, save = True, rerun=False):
    
    filename = './data/UCB_N1_T{}_alpha{}_Run{}_BerExp{}'.format(max_iter, alpha, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running UCB...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))
             
        for run in range(runs):
            if(run%200 == 0):
                print('Execution number: {}'.format(run))
            mab_ucb = UCB(alpha, theta) # reinitialize the strategy
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


def run_singleAgent_BayesUCB(max_iter, runs, c, alpha_vec, beta_vec, theta, exp_num, save = True, rerun=False):
    
    rho = 1
    filename = './data/BayesUCB_N1_T{}_Run{}_BerExp{}'.format(max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Bayes UCB...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))
     
        for run in range(runs):
            if(run%200 == 0):
                print('Execution number: {}'.format(run))
            mab_bayes_ucb = Bayes_UCB(rho, c, theta, alpha_vec, beta_vec, max_iter) # reinitialize the strategy
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

def run_singleAgent_TS(max_iter, runs, alpha_vec, beta_vec, theta, exp_num, save = True, rerun=False):
    
    rho = 1
    filename = './data/TS_N1_T{}_Run{}_BerExp{}'.format(max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Thompson Sampling...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))
     
        for run in range(runs):
            if(run%200 == 0):
                print('Execution number: {}'.format(run))
            mab_ts = Thompson_sampling(rho, theta, alpha_vec, beta_vec) # reinitialize the strategy
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

def run_MultiAgent_TS(max_iter, runs, W, alpha_vec, beta_vec, theta, exp_num, type_W ='cycle', save = True, rerun=False):
    
    N = W.shape[0]
    rho = N
    filename = './data/TS_N{}_{}_T{}_Run{}_BerExp{}'.format(N, type_W, max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Multi-agent Thompson Sampling...') 
        
        num_arms = len(theta)
        alpha_mat = np.zeros((N, num_arms))
        beta_mat = np.zeros((N, num_arms))
        
        avg_cumm_regret = np.zeros((max_iter+1))
        
        for run in range(runs):
            
            if(run%100 == 0):
                print('Execution number:', run)
            
            # reinitialize the strategy
            agent_list = []
            for i in range(N):
                curr_class = Thompson_sampling(rho, theta, alpha_vec, beta_vec)
                agent_list.append(curr_class)
                
            netwk_regret = [0]
            for t in range(max_iter):
                
                #if t%500 == 0:
                #    print('Iteration number is {}'.format(t))
                
                curr_netwk_regret = 0
                for i in range(N):
                    curr_regret = agent_list[i].play()
                    curr_netwk_regret += curr_regret
                    
                    # update the alpha and beta matrix to be used later for merging
                    alpha_mat[i, :] = deepcopy(agent_list[i].alpha_vec)
                    beta_mat[i, :] = deepcopy(agent_list[i].beta_vec)
    
                # Merge marginal posteriors at each agent
                alpha_mat = np.array(alpha_mat)
                beta_mat = np.array(beta_mat)
                
                merged_alpha = np.dot(W, alpha_mat)
                merged_beta = np.dot(W, beta_mat)
                for i in range(N):
                    agent_list[i].alpha_vec = merged_alpha[i, :]
                    agent_list[i].beta_vec = merged_beta[i, :]
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


def run_MultiAgent_BayesUCB(max_iter, runs, W, c, alpha_vec, beta_vec, theta, exp_num, type_W ='cycle',save = True, rerun=False):
    
    N = W.shape[0]
    rho = N
    filename = './data/BayesUCB_N{}_{}_T{}_Run{}_BerExp{}'.format(N, type_W, max_iter, runs, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Multi-agent Bayes UCB')
        
        
        num_arms = len(theta)
        alpha_mat = np.zeros((N, num_arms))
        beta_mat = np.zeros((N, num_arms))
        
        avg_cumm_regret = np.zeros((max_iter+1))
        
        for run in range(runs):
            
            print('Execution number:', run)
            
            # reinitialize the strategy
            agent_list = []
            for i in range(N):
                curr_class = Bayes_UCB(rho, c, theta, alpha_vec, beta_vec, max_iter)
                agent_list.append(curr_class)
                
            netwk_regret = [0]
            for t in range(max_iter):
                
                if t%500 == 0:
                    print('Iteration number is {}'.format(t))
                
                curr_netwk_regret = 0
                for i in range(N):
                    curr_regret = agent_list[i].play()
                    curr_netwk_regret += curr_regret
                    
                    # update the alpha and beta matrix to be used later for merging
                    alpha_mat[i,:] = deepcopy(agent_list[i].alpha_vec)
                    beta_mat[i,:] = deepcopy(agent_list[i].beta_vec)
    
                # Merge marginal posteriors at each agent
                alpha_mat = np.array(alpha_mat)
                beta_mat = np.array(beta_mat)
                
                merged_alpha = np.dot(W, alpha_mat)
                merged_beta = np.dot(W, beta_mat)
                for i in range(N):
                    agent_list[i].alpha_vec = merged_alpha[i,:]
                    agent_list[i].beta_vec = merged_beta[i,:]
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