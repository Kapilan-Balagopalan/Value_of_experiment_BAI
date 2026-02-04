import numpy as np

from SequentialHalving import *
from dataclasses import dataclass


class SequentialHalvingOptionalElimination:
    """
    FIXME: wait, the implementation is weird... why are we taking avg_reward in update()??
    """

    def __init__(self, K, B, nMC, reuse=False, seed=19):
        """
        B: n in the bandit book; the sampling budget.
        K: k in the bandit book; the size of the arm set
        """
        self.nMC = nMC
        self.K = K
        self.B = B
        self.reuse = reuse
        self.seed = seed
        self.epsilon = 0.01

        self.cur_best_arm = -1
        self.n_pulls = np.zeros(K, dtype=int)
        # self.avg_rewards = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.my_rng = np.random.RandomState(seed)
        self.t = 1

        # - precompute the arm pull schedule
        #print(B)
        self.T_ary, self.A_ary = SequentialHalving.calc_schedule(K, B)
        self.halving_times = np.cumsum(self.T_ary * self.A_ary[:-1])
        self.halving_times[-1] = self.B  # this is cheating, but will ensure we use all the budget.
        self.L = len(self.T_ary)
        self.ell = 1  # iteration count arm size

        self.hll = 1 # iteration count of stopping times
        self.surviving = np.arange(K)


        # - for bayesian updates
        self.prior_means = 0.5*np.ones(K)
        self.prior_vars = 0.5*np.ones(K)
        self.true_vars = np.ones(K)

        self.pos_means = 0.5 * np.ones(K)
        self.pos_vars = 0.5 * np.ones(K)

        pass

    @staticmethod
    def calc_minimum_B(K):
        L = int(np.ceil(np.log2(K)))
        return L * 2 ** L

    @staticmethod
    def calc_schedule(K, B):
        """
        In : SequentialHalving.calc_schedule(10,800)
        Out: (array([ 20,  40,  66, 100]), array([10,  5,  3,  2, 1]))
        """
        A = K
        L = int(np.ceil(np.log2(K)))
        assert (B >= SequentialHalving.calc_minimum_B(K)), "insufficient budget B"
        T_ary = np.zeros(L, dtype=int)
        A_ary = np.zeros(L + 1, dtype=int)
        for ell in range(L):
            T_ary[ell] = int(np.floor(B / (L * A)))
            A_ary[ell] = A
            A = int(np.ceil(A / 2))
        assert A == 1, "implementation error"
        A_ary[-1] = 1
        return T_ary, A_ary

    def next_arm(self):
        my_n_pulls = self.n_pulls[self.surviving]
        my_idx = np.argmin(my_n_pulls)
        idx = self.surviving[my_idx]
        return idx

    def update(self, idx, reward):
        """ if self.ell == 1+self.L, we are just receiving extra arm pull... we could use those samples somehow, but here we just ignore them and those samples will not affect calc_best_arm()
        """
        self.n_pulls[idx] += 1
        self.sum_rewards[idx] += reward
        if (self.ell <= self.L):
            # - at this point, self.t is equal to sum(self.n_pulls)
            # assert self.t == self.n_pulls.sum(), "implementation error"

            if (self.t == self.halving_times[self.hll - 1]):

                self.hll = self.hll + 1
                t_B = self.B - self.t

                me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]

                # Bayesian updates
                post_updates = bayesian_update(self.prior_means[self.surviving], self.prior_vars[self.surviving],
                                               self.true_vars[self.surviving], me, self.n_pulls[self.surviving])
                self.pos_means[self.surviving] = post_updates[0]
                self.pos_vars[self.surviving] = post_updates[1]

                # simulations

                full_A = self.A_ary[self.ell-1]
                A = self.A_ary[self.ell]
                self.cur_best_arm = self.surviving[np.argmax(me)]
                my_chosen = choose_topk_fair(me, A, randstream=self.my_rng)
                assert (len(my_chosen) == A)

                VoE_full = 0
                VoE_half = 0

                for j in range(self.nMC):
                    # Initialize algorithm
                    SH_algo_full = SequentialHalving(full_A, t_B)
                    SH_algo_half = SequentialHalving(A, t_B)
                    s = 1
                    while (s < t_B):
                        arm_to_pull = SH_algo_full.next_arm()
                        temp_sigma = self.true_vars[arm_to_pull]
                        reward = sample_rewards(self.pos_means[arm_to_pull], temp_sigma)
                        SH_algo_full.update(arm_to_pull, reward)
                        s = s + 1
                    emp_best = SH_algo_full.get_best_empirical_mean()
                    if(emp_best > 0):
                        VoE_full = VoE_full + emp_best

                    s = 1
                    while (s < t_B):
                        arm_to_pull = SH_algo_half.next_arm()
                        temp_sigma =  self.true_vars[my_chosen[arm_to_pull]]
                        reward = sample_rewards(self.pos_means[my_chosen[arm_to_pull]],temp_sigma)
                        SH_algo_half.update(arm_to_pull, reward)
                        s = s + 1
                    emp_best = SH_algo_half.get_best_empirical_mean()
                    if(emp_best > 0):
                        VoE_half = VoE_half + emp_best


                # Optional Elimination
                avg_VoE_half = VoE_half/self.nMC
                avg_VoE_full = VoE_full/self.nMC
                if (avg_VoE_half > avg_VoE_full + self.epsilon):
                    elim_flag = True
                else:
                    elim_flag = False


                # If eliminated

                if (elim_flag == True):
                    self.surviving = self.surviving[my_chosen]  # index translation

                    #ipdb.set_trace()
                    #print("Came here")
                    self.ell += 1
                    if (self.ell <= self.L and self.reuse == False):
                        self.n_pulls[self.surviving] = 0
                        self.sum_rewards[self.surviving] = 0.0
        self.t += 1

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        if np.any(self.n_pulls == 0):
            return self.cur_best_arm

        me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
        max_reward = me.max()
        return self.surviving[self.my_rng.choice(np.where(me == max_reward)[0])]

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
        return me.max()



def bayesian_update(mu_prior, var_prior, var_obs, mu_obs, n_obs):

    alpha = (n_obs*var_prior)/(var_obs + n_obs*var_prior)
    mu_post = mu_prior*(1-alpha) + mu_obs*alpha
    # Update the posterior
    var_post = var_obs*var_prior/(var_obs + n_obs*var_prior)
    return mu_post, var_post

def sample_rewards(mean, var):
    return np.random.normal(mean, var, size=1)