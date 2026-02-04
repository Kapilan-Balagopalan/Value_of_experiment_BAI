
from kjunutils3_v2 import *

class SequentialHalving:
    """
    FIXME: wait, the implementation is weird... why are we taking avg_reward in update()??
    """

    def __init__(self, K, B, reuse=False, seed=19):
        """
        B: n in the bandit book; the sampling budget.
        K: k in the bandit book; the size of the arm set
        """
        self.K = K
        self.B = B
        self.reuse = reuse
        self.seed = seed

        self.cur_best_arm = -1
        self.n_pulls = np.zeros(K, dtype=int)
        # self.avg_rewards = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.my_rng = np.random.RandomState(seed)
        self.t = 1

        # - precompute the arm pull schedule
        self.T_ary, self.A_ary = SequentialHalving.calc_schedule(K, B)
        self.halving_times = np.cumsum(self.T_ary * self.A_ary[:-1])
        self.halving_times[-1] = self.B  # this is cheating, but will ensure we use all the budget.
        self.L = len(self.T_ary)
        self.ell = 1  # iteration count
        self.surviving = np.arange(K)
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
        #print(B)
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

            if (self.t == self.halving_times[self.ell - 1]):
                me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
                A = self.A_ary[self.ell]

                self.cur_best_arm = self.surviving[np.argmax(me)]  # FIXME: not the optimal thing to do.

                my_chosen = choose_topk_fair(me, A, randstream=self.my_rng)
                assert (len(my_chosen) == A)  # just in case..
                self.surviving = self.surviving[my_chosen]  # index translation
                #print(my_chosen)
                #print(self.surviving)
                #ipdb.set_trace()
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




def choose_topk_fair(ary, k, randstream=ra):
    """
    choose top k large members from ary, but break ties uniformly at random
    """
    sidx = np.argsort(ary)[::-1]
    sary = ary[sidx]

    threshold = (sary)[k - 1]
    ties = np.where(sary == threshold)[0]
    n_ties = len(ties)
    nonties = np.where(sary > threshold)[0]
    n_nonties = len(nonties)
    if n_ties + n_nonties == k:
        chosen = sidx[:k]
    else:
        broken_ties = randstream.choice(sidx[n_nonties:n_nonties + n_ties], k - n_nonties, replace=False)
        chosen = np.concatenate((sidx[nonties], broken_ties))
    return chosen