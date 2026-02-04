from turtledemo.round_dance import stop

from SequentialHalvingOptionalElimination import *
from SequentialHalving import *
import matplotlib.pyplot as plt
import numpy as np


PRIOR_MU = 0.5
PRIOR_SIGMA = 0.5


TRUE_MU = 1
TRUE_SIGMA = 1
VAR_OBS = 1

K = 8
HORIZON = 1500



def sample_arm_means():
    #np.random.seed(42)
    # Is this necessary, or can i just manually set K arms.
    true_means = np.random.normal(TRUE_MU, TRUE_SIGMA**2, size=K)
    return true_means



def set_variance():
    return VAR_OBS*np.ones(K)








def trial(SHOE_algo,true_means, true_vars, HORIZON):
    t = 1
    while (t < HORIZON):
        arm_to_pull = SHOE_algo.next_arm()
        temp_sigma = true_vars[arm_to_pull]
        reward = sample_rewards(true_means[arm_to_pull], temp_sigma**2)
        SHOE_algo.update(arm_to_pull, reward)
        t = t + 1

    emp_best = SHOE_algo.get_best_arm()
    if true_means[emp_best] > 0 :
        return true_means[emp_best]
    else:
        return 0




#nMC = 10

true_means = sample_arm_means()
true_vars = set_variance()

nTrials = 30

n_MC_set = [1,10,20,50,100]

Y_SHOE = []
Y_SH = []

for nMC in n_MC_set:
    VoE_SHOE = 0
    VoE_SH = 0
    print("nMC", nMC)
    true_means = sample_arm_means()
    for i in range(nTrials):
        SH_algo = SequentialHalving(K, HORIZON)
        SHOE_algo = SequentialHalvingOptionalElimination(K, HORIZON, nMC)

        VoE_SH = VoE_SH + trial(SH_algo, true_means, true_vars, HORIZON)
        VoE_SHOE = VoE_SHOE + trial(SHOE_algo, true_means, true_vars, HORIZON)

    avg_VoE_SHOE= VoE_SHOE/nTrials
    avg_VoE_SH= VoE_SH/nTrials

    print("VoE_SHOE", avg_VoE_SHOE)
    print("VoE_SH", avg_VoE_SH)

    Y_SH.append(avg_VoE_SH)
    Y_SHOE.append(avg_VoE_SHOE)

plt.plot(n_MC_set, Y_SH, label="SH")
plt.plot(n_MC_set, Y_SHOE, label="SHOE")
plt.legend()
plt.show()