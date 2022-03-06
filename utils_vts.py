import numpy as np
import copy
from scipy.stats import wishart
from scipy.special import digamma
import os
import pickle
import istarmap
import tqdm
import multiprocessing
import istarmap
import math


def read_regret(filename):
    fileObject = open(filename,'rb')
    return pickle.load(fileObject)


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

class Variational_Thompson_sampling:
    def __init__(self, rho, max_iter, mu_gt, sigma_gt, pi_gt, tau_max=10):
        # K - actions or number of arms
        # M - number of guassian components in each arm
        self.rho = rho
        self.T = max_iter
        self.mu_gt = copy.deepcopy(mu_gt)  # M x K matrix
        self.sigma_gt = copy.deepcopy(sigma_gt)  # M x K matrix
        self.pi_gt = copy.deepcopy(pi_gt)  # M x K matrix
        self.num_arms = self.mu_gt.shape[1]
        self.num_comp = self.mu_gt.shape[0]
        self.curr_arm_mean = np.zeros((self.num_arms))
        self.optimal_arm_mean = np.max(np.sum(np.multiply(self.pi_gt, self.mu_gt), axis=0))
        self.tau_max = tau_max

        # Priors
        self.alpha_prior = 10*np.ones((self.num_comp, self.num_arms))
        # Alpha values < 1 makes some of the pi values zero
        self.m_prior = np.random.normal(0, 1, (self.num_comp, self.num_arms))
        # print(f'm_prior:\n {self.m_prior}')
        self.beta_prior = np.ones(((self.num_comp, self.num_arms)))  # Check how to initialize
        self.V_prior = np.ones(((self.num_comp, self.num_arms)))  # Check how to initialize
        self.nu_prior = np.ones(((self.num_comp, self.num_arms)))  # Check how to initialize

        # Estimated values
        self.alpha_es = copy.deepcopy(self.alpha_prior)
        self.m_es = copy.deepcopy(self.m_prior)
        self.beta_es = copy.deepcopy(self.beta_prior)
        self.V_es = copy.deepcopy(self.V_prior)
        self.nu_es = copy.deepcopy(self.nu_prior)

        # Estimated values of ground truth
        self.mu_es = np.ones(((self.num_comp, self.num_arms)))  # Check how to initialize
        self.sigma_es = np.ones(((self.num_comp, self.num_arms)))  # Check how to initialize
        self.pi_es = np.ones((self.num_comp, self.num_arms))  # Check later whether to initialize to ones or zeros

        self.curr_reward = None
        self.curr_arm = None
        self.num_plays = 0  # Total number of plays
        self.debug_flag = False

        self.rewards = [[] for k in range(self.num_arms)]
        # self.rewards[0] = [1.9, 6.7]
        # self.rewards[1] = [-14.8, 0.7, 5.1]
        

        # Presampling rewards
        # for k in range(0, self.num_arms):
            # self.Z_oh[:, :, k] = np.random.multinomial(1, self.pi_gt[:, k], self.T)


    def reinitialize(self):
        self.rewards = [[] for k in range(self.num_arms)]
        self.num_plays = 0  # Total number of plays
        self.alpha_es = copy.deepcopy(self.alpha_prior)
        self.m_es = copy.deepcopy(self.m_prior)
        self.beta_es = copy.deepcopy(self.beta_prior)
        self.V_es = copy.deepcopy(self.V_prior)
        self.nu_es = copy.deepcopy(self.nu_prior)


    # Return the action
    def select_action(self):
        # if (self.debug_flag == True):
        #     print(self.num_plays)
        # if (self.debug_flag == True):
        #     __import__("pdb").set_trace()

        self.nu_es[self.nu_es < 1] = 1.0
        for k in range(self.num_arms):
            self.pi_es[:, k] = np.random.dirichlet(self.alpha_es[:, k], 1)
            # assert(np.abs(1.0 - np.sum(self.pi_es[:, k])) < 0.1)
            for m in range(self.num_comp):
                precision = wishart.rvs(self.nu_es[m, k], self.V_es[m, k])
                self.sigma_es[m, k] = 1 / (self.beta_es[m, k] * precision)
                self.mu_es[m, k] = np.random.normal(self.m_es[m, k], self.sigma_es[m, k])
            self.curr_arm_mean[k] = np.dot(self.pi_es[:, k], self.mu_es[:, k])
        self.curr_arm = np.argmax(self.curr_arm_mean)
        # print(f'Arm: {self.curr_arm}')
        
    # Get reward and regret for the choosen action
    def get_reward_regret(self):
        Z_oh = np.random.multinomial(1, self.pi_gt[:, self.curr_arm])
        self.gaussian_mixture_gt = np.where(Z_oh)
        # print(f'Z_oh: {self.Z_oh[self.num_plays, :, self.curr_arm]}')
        # print(f'Gaussian mixture component: {self.gaussian_mixture_gt}')
        self.curr_reward = np.random.normal(self.mu_gt[self.gaussian_mixture_gt, self.curr_arm], self.sigma_gt[self.gaussian_mixture_gt, self.curr_arm])
        self.rewards[self.curr_arm].append(self.curr_reward[0, 0])
        # print(f'Reward: {self.curr_reward}')
        regret = self.optimal_arm_mean - self.curr_reward
        self.num_plays += 1  # update the total number of plays
        return regret
        
    def update(self):
        def step_1():
            '''
            $\log \gamma_{k,m,t}(\tau) = \psi(\alpha_{k,m}(\tau-1))- \psi\left(\sum_{m' =1}^M \alpha_{k,m'}(\tau-1)\right) + \frac{1}{2}\psi\left(\frac{\nu_{k,m}(\tau-1)}{2}\right)-\frac{\nu_{k,m}(\tau-1)}{2}(r_{k,t}-m_{k,m}(\tau-1))^2 V_{k,m}(\tau-1)+ \log 2 + \log |V_{k,m}(\tau-1)|-\frac{1}{2\beta_{k,m}(\tau-1)} + \text{constant}$
            
            with $\sum_{m=1}^M \gamma_{k,m,t}(\tau) = 1$ for all $k,t$ and where $\psi(\cdot)$ is the digamma function.
            '''
            # Update gamma for each k, m, num_plays.
            # if (self.debug_flag == True and self.curr_arm == 4):
            #     __import__("pdb").set_trace()

            num_plays_arm = len(self.rewards[self.curr_arm])
            reward_k = np.tile(np.array(self.rewards[self.curr_arm]).reshape(1, -1), (self.num_comp, 1))  # DONE. Check that all rows are same and dimensions = [num_comp, num_plays_arm]
            alpha_k = np.tile(self.alpha_es[:, self.curr_arm].reshape(-1, 1), (1, num_plays_arm))
            V_k = np.tile(self.V_es[:, self.curr_arm].reshape(-1, 1), (1, num_plays_arm))
            m_k = np.tile(self.m_es[:, self.curr_arm].reshape(-1, 1), (1, num_plays_arm))
            nu_k = np.tile(self.nu_es[:, self.curr_arm].reshape(-1, 1), (1, num_plays_arm))
            beta_k = np.tile(self.beta_es[:, self.curr_arm].reshape(-1, 1), (1, num_plays_arm))
            term_1 = digamma(alpha_k)
            term_2 = -np.tile(digamma(np.sum(alpha_k, axis=0)), (self.num_comp, 1))  # DONE. Check that all entries are the same
            term_3 = 0.5 * digamma(nu_k / 2)
            term_4_1 = - (nu_k / 2)
            term_4_2 = np.multiply(reward_k - m_k, reward_k - m_k)
            term_4_3 = V_k
            term_4 = np.multiply(np.multiply(term_4_1, term_4_2), term_4_3)
            term_5 = np.log(2 * np.ones((self.num_comp, num_plays_arm)))
            term_6 = np.log(np.abs(V_k))  
            term_7 = - 1 / (2 * beta_k)
            log_gamma_k_t = self.rho*(term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7)
            gamma_k_t = np.exp(log_gamma_k_t)
            gamma_k_t = np.divide(gamma_k_t, np.tile(np.sum(gamma_k_t, axis=0), (self.num_comp, 1)))  # DONE. Check that all columns add up to 1
            # print(alpha_k)
            # print(V_k)
            # print(m_k)
            # print(term_1)
            # print(term_2)
            # print(term_3)
            # print(term_4)
            # print(term_5)
            # print(term_6)
            # print(term_7)
            # print(f'gamma_k_t: {gamma_k_t}')  # Check that each element is not always 1/self.num_comp
            return gamma_k_t

        def step_2(gamma):
            '''
            $n_{k,m}(\tau) = \sum_{t=1}^{n}\gamma_{k,m,t}(\tau)$

            $\overline{r}_{k,m}(\tau) = \frac{1}{n_{k,m}(\tau)}\sum_{t=1}^{n} \gamma_{k,m,t}(\tau)r_{k,t}$
            
            $S_k(\tau) = \frac{1}{n_{k,m}(\tau)}\sum_{t=1}^{n}\gamma_{k,m,t}(\tau)(r_{k,t}-\overline{r}_{k,m}(\tau))^2$
            '''
            num_plays_arm = len(self.rewards[self.curr_arm])
            reward_k = np.tile(np.array(self.rewards[self.curr_arm]).reshape(1, -1), (self.num_comp, 1))  # DONE. Check that all rows are same and dimensions = [num_comp, num_plays_arm]

            # if(self.debug_flag == True):
            #     __import__("pdb").set_trace()

            n_k = self.rho*np.sum(gamma, axis=1)
            indices = np.where(n_k < 1e-5)
            # print(f'reward history: {self.reward_array}')
            r_bar_k = self.rho*np.divide(np.sum(np.multiply(gamma, reward_k), axis=1), n_k)
            r_difference = reward_k - np.tile(r_bar_k.reshape(-1, 1), (1, num_plays_arm))  # Check that this is num_comp x self.rewards[self.curr_arm]
            S_k = self.rho*np.divide(np.sum(np.multiply(gamma, np.multiply(r_difference, r_difference)), axis=1), n_k)
            # print(n_k)
            # print(r_bar_k)
            # print(S_k)  # Check that this is not always zero

            '''
            $\alpha_{k,m}(\tau) = \alpha_{k,m}(0) + n_{k,m}(\tau)$
            
            $\beta_{k,m}(\tau) = \beta_{k,m}(0) + n_{k,m}(\tau)$
            
            $m_{k,m}(\tau) = \frac{1}{\beta_{k,m}(\tau)}(\beta_{k,m}(0)m_{k,m}(0) + n_{k,m}(\tau)\overline{r}_{k,m}(\tau))$
            
            $V^{-1}_{k,m}(\tau) = V^{-1}_{k,m}(0) + n_k(\tau) S_{k,m}(\tau) + \frac{\beta_{k,m}(0) n_{k,m}(\tau)}{\beta_{k,m}(0)+ n_{k,m}(\tau)}(\overline{r}_{k,m}(\tau)-m_{k,m}(0))^2$
            
            $\nu_{k,m}(\tau) = \nu_{k,m}(0) + n_{k,m}(\tau)$     
            '''
            self.alpha_es[:, self.curr_arm] = self.alpha_prior[:, self.curr_arm] + n_k
            self.beta_es[:, self.curr_arm] = self.beta_prior[:, self.curr_arm] + n_k
            term_1 = self.beta_es[:, self.curr_arm]
            term_2 = np.multiply(self.beta_prior[:, self.curr_arm], self.m_prior[:, self.curr_arm])
            term_3 = np.multiply(n_k, r_bar_k)
            term_3[indices] = 0
            self.m_es[:, self.curr_arm] = np.divide(term_2 + term_3, term_1)
            term_1 = 1 / self.V_prior[:, self.curr_arm]
            term_2 = np.multiply(n_k, S_k)
            term_2[indices] = 0
            term_3_a = np.divide(np.multiply(self.beta_prior[:, self.curr_arm], n_k), self.beta_prior[:, self.curr_arm] + n_k)
            term_3_b = np.multiply(r_bar_k - self.m_prior[:, self.curr_arm], r_bar_k - self.m_prior[:, self.curr_arm])
            term_3 = np.multiply(term_3_a, term_3_b)
            term_3[indices] = 0
            self.V_es[:, self.curr_arm] = 1 / (term_1 + term_2 + term_3)  # divide needs to be replaced with inverse. parameter of divide is a vector. How do we compute inverse?
            self.nu_es[:, self.curr_arm] = self.nu_prior[:, self.curr_arm] + n_k
            # print(self.alpha_es[:, self.curr_arm])
            # print(self.beta_es[:, self.curr_arm])
            # print(self.m_es[:, self.curr_arm])
            # print(self.V_es[:, self.curr_arm])
            # print(self.nu_es[:, self.curr_arm])

        for tau in range(self.tau_max):
            gamma = step_1()
            step_2(gamma)
            
    # Play one round
    def play(self):
        self.select_action()  # select action
        regret = self.get_reward_regret()  # observe reward for the selected action
        self.update()  # update the model
        return regret


def run_singleAgent_VTS(max_iter, runs, mu_gt, sigma_gt, pi_gt, exp_num, tau_max=10, save = True, rerun=False):
    rho = 1.0
    filename = './data/VTS_N1_T{}_Run{}_Tau{}_GaussianMixtureExp{}'.format(max_iter, runs, tau_max, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Variational Thompson Sampling...') 
        
        avg_cumm_regret = np.zeros((max_iter+1))

        for curr_run in tqdm.tqdm(range(runs)):
            mab_vts = Variational_Thompson_sampling(rho, max_iter, mu_gt, sigma_gt, pi_gt, tau_max)  # reinitialize the strategy
            regret = [0]
            for t in range(max_iter):
                curr_regret = mab_vts.play()
                regret.append(regret[-1] + curr_regret[0, 0])
            regret = np.array(regret)
            avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(curr_run+1)) + np.multiply(regret, 1/(curr_run+1))
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


def run_MultiAgent_VTS_per_run(max_iter, W, mu_gt, sigma_gt, pi_gt, exp_num, tau_max=10, update_frequency=10, type_W ='cycle', save = True, rerun=False):
    N = W.shape[0]
    rho = N
    np.random.seed()

    [num_comp, num_arms] = mu_gt.shape
    alpha_mat = np.zeros((N, num_comp, num_arms))
    nu_mat = np.zeros((N, num_comp, num_arms))
    V_mat = np.zeros((N, num_comp, num_arms))
    beta_mat = np.zeros((N, num_comp, num_arms))
    m_mat = np.zeros((N, num_comp, num_arms))
    agent_list = []
    for i in range(N):
        curr_class = Variational_Thompson_sampling(rho, max_iter, mu_gt, sigma_gt, pi_gt, tau_max)  # reinitialize the strategy
        agent_list.append(curr_class)
        # print('At initialization \n')
        # print('Priors \n')
        # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_prior}')
        # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_prior}')
        # print(f'V_matrix of agent {i}: \n{agent_list[i].V_prior}')
        # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_prior}')
        # print(f'm_matrix of agent {i}: \n{agent_list[i].m_prior}')

        # print('Estimated \n')
        # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_es}')
        # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_es}')
        # print(f'V_matrix of agent {i}: \n{agent_list[i].V_es}')
        # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_es}')
        # print(f'm_matrix of agent {i}: \n{agent_list[i].m_es}')

    netwk_regret = [0]
    for t in range(max_iter):
        # print('Iteration number is {}'.format(t))
        '''
        if(t%500 == 0):
            print('Iteration number is {}'.format(t))
        '''
        curr_netwk_regret = 0

        for i in range(N):
            # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_es}')

            # print(f'i: {i}')
            curr_regret = agent_list[i].play()
            curr_netwk_regret += curr_regret[0, 0]

            # update the alpha and beta matrix to be used later for merging
            if (t % update_frequency == 0 and t != 0):
                # print('Before merging \n')
                # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_es}')
                # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_es}')
                # print(f'V_matrix of agent {i}: \n{agent_list[i].V_es}')
                # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_es}')
                # print(f'm_matrix of agent {i}: \n{agent_list[i].m_es}')
                alpha_mat[i, :, :] = agent_list[i].alpha_es
                nu_mat[i, :, :] = agent_list[i].nu_es
                V_mat[i, :, :] = agent_list[i].V_es
                beta_mat[i, :, :] = agent_list[i].beta_es
                m_mat[i, :, :] = agent_list[i].m_es

        if (t % update_frequency == 0 and t != 0):
            # Merge marginal posteriors at each agent
            # W_2 = np.transpose(np.tile(np.reshape(W, (N, N, 1, 1)), (1, 1, M, K)), (1,0,2,3))
            # A_2 = np.transpose(np.tile(A, (N, 1, 1, 1)), (1,0,2,3)
            W_4D = np.tile(np.reshape(W, (N, N, 1, 1)), (1, 1, num_comp, num_arms))
            alpha_mat_4D = np.tile(alpha_mat, (N, 1, 1, 1))
            alpha_merged_mat = np.sum(np.multiply(W_4D, alpha_mat_4D), axis=1)
            nu_mat_4D = np.tile(nu_mat, (N, 1, 1, 1))
            nu_merged_mat = np.sum(np.multiply(W_4D, nu_mat_4D), axis=1)
            V_mat_4D = np.reciprocal(np.tile(V_mat, (N, 1, 1, 1)))
            V_merged_mat = np.reciprocal(np.sum(np.multiply(W_4D, V_mat_4D), axis=1))
            beta_mat_4D = np.tile(beta_mat, (N, 1, 1, 1))
            beta_merged_mat = np.sum(np.multiply(W_4D, beta_mat_4D), axis=1)
            m_mat_4D = np.tile(m_mat, (N, 1, 1, 1))
            m_merged_mat = np.multiply(np.sum(np.multiply(np.multiply(W_4D, m_mat_4D), beta_mat_4D), axis=1), np.reciprocal(beta_merged_mat))

            for i in range(N):
                agent_list[i].alpha_prior = alpha_merged_mat[i, :, :]
                agent_list[i].nu_prior = nu_merged_mat[i, :, :]
                agent_list[i].V_prior = V_merged_mat[i, :, :]
                agent_list[i].beta_prior = beta_merged_mat[i, :, :]
                agent_list[i].m_prior = m_merged_mat[i, :, :]
                agent_list[i].reinitialize()
                # print('After merging \n')
                # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_prior}')
                # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_prior}')
                # print(f'V_matrix of agent {i}: \n{agent_list[i].V_prior}')
                # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_prior}')
                # print(f'm_matrix of agent {i}: \n{agent_list[i].m_prior}')
                # print('\n\n\n')
                # __import__("pdb").set_trace()

        # print(netwk_regret)
        netwk_regret.append(netwk_regret[-1] + curr_netwk_regret/N)
    return netwk_regret


def run_MultiAgent_VTS_parallel(max_iter, runs, W, mu_gt, sigma_gt, pi_gt, exp_num, tau_max=10, update_frequency=10, type_W ='cycle', save = True, rerun=False):
    N = W.shape[0]
    filename = './data/VTS_N{}_T{}_W{}_Run{}_Tau{}_Freq{}_GaussianMixtureExp{}'.format(N, max_iter, type_W, runs, tau_max, update_frequency, exp_num)

    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Multi-agent Variational_Thompson Sampling...')

        exp_list = [[max_iter, W, mu_gt, sigma_gt, pi_gt, exp_num, tau_max, update_frequency, type_W, save, rerun] for run in range(runs)]
        with multiprocessing.Pool(min(12, multiprocessing.cpu_count())) as pool:
            all_netwk_regrets = pool.starmap(run_MultiAgent_VTS_per_run, exp_list)
        all_netwk_regrets = np.array(all_netwk_regrets)
        avg_cumm_regret = np.mean(all_netwk_regrets, axis=0)

        # avg_cumm_regret = np.zeros((max_iter+1))
        # queue = multiprocessing.Queue()
        # processes = []
        # for run in tqdm.tqdm(range(runs)):
        # for run in range(runs):
        #     p = multiprocessing.Process(target=run_MultiAgent_VTS_per_run, args=(queue, max_iter, W, mu_gt, sigma_gt, pi_gt, exp_num, tau_max, update_frequency, type_W, save, rerun))
        #     processes.append(p)
        #     p.start()
        # # for run, p in enumerate(processes):
        # for run, p in enumerate(tqdm.tqdm(processes)):
        #     netwk_regret = queue.get()
        #     avg_cumm_regret = np.multiply(avg_cumm_regret, 1-1/(run+1)) + np.multiply(netwk_regret, 1/(run+1))
        # for p in processes:
        #     p.join()
        if save:
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fileObject = open(filename,'wb')
            pickle.dump(avg_cumm_regret,fileObject)
            fileObject.close()
            print('Saved')
    return avg_cumm_regret    


def run_MultiAgent_VTS(max_iter, runs, W, mu_gt, sigma_gt, pi_gt, exp_num, tau_max=10, update_frequency=10, type_W ='cycle', save = True, rerun=False):
    N = W.shape[0]
    rho = N
    np.random.seed(0)
    filename = './data/VTS_N{}_T{}_Run{}_Tau{}_Freq{}_GaussianMixtureExp{}'.format(N, max_iter, runs, tau_max, update_frequency, exp_num)
    
    if not rerun and os.path.isfile(filename):
        print('Saved file was loaded')
        avg_cumm_regret = read_regret(filename)
    else:
        print('Running Multi-agent Variational_Thompson Sampling...') 
        
        [num_comp, num_arms] = mu_gt.shape
        avg_cumm_regret = np.zeros((max_iter+1))

        for run in tqdm.tqdm(range(runs)):
            alpha_mat = np.zeros((N, num_comp, num_arms))
            nu_mat = np.zeros((N, num_comp, num_arms))
            V_mat = np.zeros((N, num_comp, num_arms))
            beta_mat = np.zeros((N, num_comp, num_arms))
            m_mat = np.zeros((N, num_comp, num_arms))
            # print('Execution number: {}'.format(run))
            # if(run%100 == 0):
            #     print('Execution number: {}'.format(run))
            # reinitialize the strategy
            agent_list = []
            for i in range(N):
                curr_class = Variational_Thompson_sampling(rho, max_iter, mu_gt, sigma_gt, pi_gt, tau_max)  # reinitialize the strategy
                agent_list.append(curr_class)
                # print('At initialization \n')
                # print('Priors \n')
                # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_prior}')
                # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_prior}')
                # print(f'V_matrix of agent {i}: \n{agent_list[i].V_prior}')
                # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_prior}')
                # print(f'm_matrix of agent {i}: \n{agent_list[i].m_prior}')

                # print('Estimated \n')
                # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_es}')
                # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_es}')
                # print(f'V_matrix of agent {i}: \n{agent_list[i].V_es}')
                # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_es}')
                # print(f'm_matrix of agent {i}: \n{agent_list[i].m_es}')

            netwk_regret = [0]
            for t in range(max_iter):
                # if (run == 44):
                #     print('Iteration number is {}'.format(t))
                '''
                if(t%500 == 0):
                    print('Iteration number is {}'.format(t))
                '''
                curr_netwk_regret = 0

                for i in range(N):
                    # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_es}')
                    # if (run == 44 and t == 17):
                    #     print(f'Agent: {i}')
                    # if (run == 44 and t == 17 and i == 8):
                    # if (run == 44 and t == 21 and i == 2):
                        # print(f't: {t} agent: {i}')
                        # agent_list[i].debug_flag = True
                        # __import__("pdb").set_trace()

                    # print(f'i: {i}')
                    curr_regret = agent_list[i].play()
                    curr_netwk_regret += curr_regret[0,0]

                    # update the alpha and beta matrix to be used later for merging
                    if (t % update_frequency == 0 and t != 0):
                        # print('Before merging \n')
                        # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_es}')
                        # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_es}')
                        # print(f'V_matrix of agent {i}: \n{agent_list[i].V_es}')
                        # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_es}')
                        # print(f'm_matrix of agent {i}: \n{agent_list[i].m_es}')
                        alpha_mat[i, :, :] = agent_list[i].alpha_es
                        nu_mat[i, :, :] = agent_list[i].nu_es
                        V_mat[i, :, :] = agent_list[i].V_es
                        beta_mat[i, :, :] = agent_list[i].beta_es
                        m_mat[i, :, :] = agent_list[i].m_es
                
                if (t % update_frequency == 0 and t != 0):
                    # Merge marginal posteriors at each agent
                    # W_2 = np.transpose(np.tile(np.reshape(W, (N, N, 1, 1)), (1, 1, M, K)), (1,0,2,3))
                    # A_2 = np.transpose(np.tile(A, (N, 1, 1, 1)), (1,0,2,3)
                    W_4D = np.tile(np.reshape(W, (N, N, 1, 1)), (1, 1, num_comp, num_arms))
                    alpha_mat_4D = np.tile(alpha_mat, (N, 1, 1, 1))
                    alpha_merged_mat = np.sum(np.multiply(W_4D, alpha_mat_4D), axis=1)
                    nu_mat_4D = np.tile(nu_mat, (N, 1, 1, 1))
                    nu_merged_mat = np.sum(np.multiply(W_4D, nu_mat_4D), axis=1)
                    V_mat_4D = np.reciprocal(np.tile(V_mat, (N, 1, 1, 1)))
                    V_merged_mat = np.reciprocal(np.sum(np.multiply(W_4D, V_mat_4D), axis=1))
                    beta_mat_4D = np.tile(beta_mat, (N, 1, 1, 1))
                    beta_merged_mat = np.sum(np.multiply(W_4D, beta_mat_4D), axis=1)
                    m_mat_4D = np.tile(m_mat, (N, 1, 1, 1))
                    # m_merged_mat = np.multiply(np.sum(np.multiply(W_4D, m_mat_4D), axis=1), np.reciprocal(beta_merged_mat))
                    m_merged_mat = np.multiply(np.sum(np.multiply(np.multiply(W_4D, m_mat_4D), beta_mat_4D), axis=1), np.reciprocal(beta_merged_mat))
                    # m_merged_mat = m_mat
                    
                    for i in range(N):
                        agent_list[i].alpha_prior = alpha_merged_mat[i, :, :]
                        agent_list[i].nu_prior = nu_merged_mat[i, :, :]
                        agent_list[i].V_prior = V_merged_mat[i, :, :]
                        agent_list[i].beta_prior = beta_merged_mat[i, :, :]
                        agent_list[i].m_prior = m_merged_mat[i, :, :]
                        agent_list[i].reinitialize()
                        # print('After merging \n')
                        # print(f'alpha_matrix of agent {i}: \n{agent_list[i].alpha_prior}')
                        # print(f'nu_matrix of agent {i}: \n{agent_list[i].nu_prior}')
                        # print(f'V_matrix of agent {i}: \n{agent_list[i].V_prior}')
                        # print(f'beta_matrix of agent {i}: \n{agent_list[i].beta_prior}')
                        # print(f'm_matrix of agent {i}: \n{agent_list[i].m_prior}')
                        # print('\n\n\n')
                        # __import__("pdb").set_trace()

                # print(netwk_regret)
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

