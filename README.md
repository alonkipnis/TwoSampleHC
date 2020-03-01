# TwoSampleHC -- Higher Criticism Test between Two Frequency Tables

An adaptation of the Donoho-Jin-Tukey Higher-Critisim (HC) test to frequency tables. This adapatation uses a binomial allocation model for the number of occurances of each feature in two samples, each of which is associated with a frequency table. An exact binomial test on each feature yields a p-value. The HC statistic is used to combine these P-values that can be uses either as a measure of similarity or to construct a test against the null hypothesis that the two tables are sampled from the same population.

This test is particularly useful in identifying non-null effects under weak and sparse alternatives, i.e., when the difference between the tables is due to few features, and the evidence each such feature provide is realtively weak. More details and applications in text classification challenges can be found in
[1] Alon Kipnis, ``Higher Criticism for Discriminating Word Frequency Tables and Testing Authorship'', 2019
[2] David Donoho and Alon Kipnis, ``Two-sample Testing for Large, Sparse High-Dimensional Multinomials under Rare and WeakPerturbations'', 2020. 

## Example:
```
import numpy as np

N = 1000 # number of features
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

print("TV distance between P and Q: ", 0.5*np.sum(np.abs(P-Q)))
print("Higher-Criticism score for testing P == Q: ", HC)  
# TV distance between P and Q:  0.11229216095188953
# Higher-Criticism score for testing P == Q:  3.874043440201504

# (HC score rarely goes above 2.5 if P == Q)
```
