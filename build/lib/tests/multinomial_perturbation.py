from TwoSampleHC import two_sample_pvals
from TwoSampleHC import hc_vals
import numpy as np

N = 10000 # number of features
n = 5 * N #number of samples

P = 1 / np.arange(1,N+1) # Zipf base distribution
P = P / P.sum()

ep = 0.03 #fraction of features to perturb
mu = 0.005 #intensity of perturbation

TH = np.random.rand(N) < ep
Q = P.copy()
Q[TH] += mu
Q = Q / np.sum(Q)

smp_P = np.random.multinomial(n, P)  # sample form P
smp_Q = np.random.multinomial(n, Q)  # sample from Q

pv = two_sample_pvals(smp_Q, smp_P) # binomial P-values
HC, p_th = hc_vals(pv, alpha = 0.25) # Higher Criticism test

self.assertTrue(np.abs(HC - 21)<3)