import numpy as np
import pandas as pd
from scipy.stats import binom, norm, poisson, beta

class HC(object):
    """
    Higher Criticism test 

    References:
    [1] Donoho, D. L. and Jin, J., "Higher criticism for detecting sparse
     hetrogenous mixtures", Annals of Stat. 2004
    [2] Donoho, D. L. and Jin, J. "Higher critcism thresholding: Optimal 
    feature selection when useful features are rare and weak", proceedings
    of the national academy of sciences, 2008.
    ========================================================================

    Args:
    -----
        pvals    list of p-values. P-values that are np.nan are exluded.
        stbl     normalize by expected P-values (stbl=True) or observed 
                 P-values (stbl=False). stbl=True was suggested in [2].
                 stbl=False in [1]. 
        gamma    lower fruction of p-values to use.
        
    Methods :
    -------
        HC       HC and P-value attaining it
        HCstar   sample adjustet HC (HCdagger in [1])
        HCjin    a version of HC from [3] (Jin & Wang 2016)
    """

    def __init__(self, pvals, stbl=True) :

        self._N = len(pvals)
        assert (self._N > 0)
        self._EPS = 1 / (1e4 + self._N ** 2)
        self._istar = 1

        self._sorted_pvals = np.sort(np.asarray(pvals.copy()))  # sorted P-values
        self._uu = np.linspace(1 / self._N, 1, self._N)
        self._uu[-1] -= self._EPS # we assume that the largest P-value
                                  # has no effect on the results
        if stbl:
            denom = np.sqrt(self._uu * (1 - self._uu))
        else:
            denom = np.sqrt(self._sorted_pvals * (1 - self._sorted_pvals))

        self._zz = np.sqrt(self._N) * (self._uu - self._sorted_pvals) / denom

        self._imin_star = np.argmax(self._sorted_pvals > (1 - self._EPS) / self._N)
        self._imin_jin = np.argmax(self._sorted_pvals > np.log(self._N) / self._N)


    def _calculateHC(self, imin, imax) :
        if imin > imax:
            return np.nan
        if imin==imax:
            self._istar = imin
        else: 
            self._istar = np.argmax(self._zz[imin:imax]) + imin
        zMaxStar = self._zz[self._istar]
        return zMaxStar, self._sorted_pvals[self._istar]

    def HC(self, gamma=0.2) : 
        """
        Higher Criticism test statistic

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Return:
        -------
        HC test score, P-value attaining it

        """
        imin = 0
        imax = np.maximum(imin, int(gamma * self._N + 0.5))        
        return self._calculateHC(imin, imax)

    def HCjin(self, gamma=0.2) :
        """sample-adjusted higher criticism score from [2]

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Return:
        -------
        HC score, P-value attaining it

        """
        
        imin = self._imin_jin
        imax = np.maximum(imin + 1,
                        int(np.floor(gamma * self._N + 0.5)))
        return self._calculateHC(imin, imax)
    
    def HCstar(self, gamma=0.2) :
        """sample-adjusted higher criticism score

        Args:
        -----
        'gamma' : lower fraction of P-values to consider

        Returns:
        -------
        HC score, P-value attaining it

        """

        imin = self._imin_star
        imax = np.maximum(imin + 1,
                        int(np.floor(gamma * self._N + 0.5)))        
        return self._calculateHC(imin, imax)

    def get_state(self):
        return {'pvals' : self._sorted_pvals, 
                'u' : self._uu,
                'z' : self._zz,
                'imin_star' : self._imin_star,
                'imin_jin' : self._imin_jin,
                }

    
def two_sample_test(smp1, smp2, data_type = 'counts',
                         alt='two-sided', **kwargs) :
    """
    Returns HC score and HC threshold in a two-sample test. 
    =========================================================

    Args:
    ----

    smp1, smp2    dataset representing samples from the identical or different
                  populations.
    data_type     either 'counts' of categorical variables or 'reals'
    alt           how to compute P-values ('two-sided' or 'greater')
    kwargs        additional arguments for the class HC and pvalue computation

    Returns:
    -------
    (HC, HCT)     HC score, HC threshold P-value

    """

    stbl = kwargs.get('stbl', True)
    randomize = kwargs.get('randomize', False)
    gamma = kwargs.get('gamma', 0.2)

    smp1 = np.array(smp1)
    smp2 = np.array(smp2)

    if data_type == 'counts' :
        pvals = two_sample_pvals(smp1, smp2, randomize=randomize, alt=alt)
    elif data_type == 'reals' :
        z = (smp1 - smp2) / np.sqrt(2)
        if alt == 'greater' :
            pvals = norm.sf(z)
        else :
            pvals = norm.sf(np.abs(z))

    return HC(pvals[~np.isnan(pvals)], stbl).HCstar(gamma)

def hc_vals(pv, gamma=0.2, minPv='one_over_n', stbl=True):
    """
    This is the old version of the HC test that is now doen using 
    the class HC. 

    Higher Criticism test 

    Args:
    -----
        pv : list of p-values. P-values that are np.nan are exluded.
        gamma : lower fruction of p-values to use.
        stbl : use expected p-value ordering (stbl=True) or observed 
                (stbl=False)
        minPv : integer or string 'one_over_n' (default).
                 Ignote smallest minPv-1 when computing HC score.

    Return :
    -------
        hc_star : sample adapted HC (HC dagger in [1])
        p_star : HC threshold: upper boundary for collection of
                 p-value indicating the largest deviation from the
                 uniform distribution.

    """
    EPS = 0.001
    pv = np.asarray(pv).copy()
    n = len(pv)  #number of features
    pv[np.isnan(pv)] = 1-EPS
    #n = len(pv)
    hc_star = np.nan
    p_star = np.nan

    if n > 1:
        ps_idx = np.argsort(pv)
        ps = pv[ps_idx]  #sorted pvals

        uu = np.linspace(1 / n, 1-EPS, n)  #expectation of p-values under
        # H0; largest P-value assumed to have not effect. 
        i_lim_up = np.maximum(int(np.floor(gamma * n + 0.5)), 1)

        ps = ps[:i_lim_up]
        uu = uu[:i_lim_up]
        
        if minPv == 'one_over_n' :
            i_lim_low = np.argmax(ps > (1-EPS)/n)
        else :
            i_lim_low = minPv

        if stbl:
            z = (uu - ps) / np.sqrt(uu * (1 - uu)) * np.sqrt(n)
        else:
            z = (uu - ps) / np.sqrt(ps * (1 - ps)) * np.sqrt(n)

        i_lim_up = max(i_lim_low + 1, i_lim_up)

        i_max_star = np.argmax(z[i_lim_low:i_lim_up]) + i_lim_low

        z_max_star = z[i_max_star]

        hc_star = z[i_max_star]
        p_star = ps[i_max_star]

    return hc_star, p_star

def binom_test(x, n, p, alt='greater') :
    """
    Returns:
    --------
    Prob(Bin(n,p) >= x) ('greater')
    or Prob(Bin(n,p) <= x) ('less')

    Note: for small values of Prob there are differences
    fron scipy.python.binom_test. It is unclear which one is 
    more accurate.
    """
    n = n.astype(int)
    if alt == 'greater' :
        return binom.sf(x, n, p) + binom.pmf(x, n, p)
    if alt == 'less' :
        return binom.cdf(x, n, p)


def binom_test_two_sided_slow(x, n, p) :
    """
     Calls scipy.stats.binom_test on each entry of
     an array. Slower than binom_test_two_sided but 
     perhaps more accurate. 
    """
    #slower
    def my_func(r) :
        from scipy.stats import binom_test
        return binom_test(r[0],r[1],r[2])

    a = np.concatenate([np.expand_dims(x,1),
                    np.expand_dims(n,1),
                    np.expand_dims(p,1)],
                    axis = 1)

    pv = np.apply_along_axis(my_func,1,a)

    return pv

def poisson_test_random(x, lmd) :
    """Prob( Pois(n,p) >= x ) + randomization """
    p_down = 1 - poisson.cdf(x, lmd)
    p_up = 1 - poisson.cdf(x, lmd) + poisson.pmf(x, lmd)
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (x != 0) + U * (x == 0)

def binom_test_two_sided(x, n, p) :
    """
    Returns:
    --------
    Prob( |Bin(n,p) - np| >= |x-np| )

    Note: for small values of Prob there are differences
    fron scipy.python.binom_test. It is unclear which one is 
    more accurate.
    """

    n = n.astype(int)

    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
        
    prob = np.minimum(p_up, 1)
    return prob * (n != 0) + 1. * (n == 0)


def binom_test_two_sided_random(x, n, p) :
    """
    Returns:
    --------
    pval  : random number such that 
            Prob(|Bin(n,p) - np| >= 
            |InvCDF(pval|Bin(n,p)) - n p|) ~ U(0,1)
    """

    x_low = n * p - np.abs(x-n*p)
    x_high = n * p + np.abs(x-n*p)

    n = n.astype(int)

    p_up = binom.cdf(x_low, n, p)\
        + binom.sf(x_high-1, n, p)
    
    p_down = binom.cdf(x_low-1, n, p)\
        + binom.sf(x_high, n, p)
    
    U = np.random.rand(x.shape[0])
    prob = np.minimum(p_down + (p_up-p_down)*U, 1)
    return prob * (n != 0) + U * (n == 0)

def binom_var_test_df(c1, c2, sym=False, max_m=-1) :
    """ Binmial variance test along stripes. 
        This version returns all sub-calculations
    Args:
    ----
    c1, c2 : list of integers represents count data from two sample
    sym : flag indicates wether the size of both sample is assumed
          identical, hence p=1/2
    """

    df_smp = pd.DataFrame({'n1' : c1, 'n2' : c2})
    df_smp.loc[:,'N'] = df_smp.agg('sum', axis = 'columns')
    
    if max_m > 0 :
        df_smp = df_smp[df_smp.n1 + df_smp.n2 <= max_m]
        
    df_hist = df_smp.groupby(['n1', 'n2']).count().reset_index()
    df_hist.loc[:,'m'] = df_hist.n1 + df_hist.n2
    df_hist = df_hist[df_hist.m > 0]
    
    df_hist.loc[:,'N1'] = df_hist.n1 * df_hist.N
    df_hist.loc[:,'N2'] = df_hist.n2 * df_hist.N

    df_hist.loc[:,'NN1'] = df_hist.N1.sum()
    df_hist.loc[:,'NN2'] = df_hist.N2.sum()

    df_hist = df_hist.join(df_hist.filter(['m', 'N1', 'N2', 'N']).groupby('m').agg('sum'),
                           on = 'm', rsuffix='_m')
    if max_m == -1 :
        df_hist = df_hist[df_hist.N_m > np.maximum(df_hist.n1, df_hist.n2)]

    df_hist.loc[:,'p'] = df_hist['NN1'] / (df_hist['NN1'] + df_hist['NN2'])

    df_hist.loc[:,'s'] = (df_hist.n1 - df_hist.m * df_hist.p) ** 2 * df_hist.N
    df_hist.loc[:,'Es'] = df_hist.N_m * df_hist.m * df_hist.p * (1 - df_hist.p)
    df_hist.loc[:,'Vs'] = 2 * df_hist.N_m *  df_hist.m * (df_hist.m) * ( df_hist.p * (1 - df_hist.p) ) ** 2
    df_hist = df_hist.join(df_hist.groupby('m').agg('sum').s, on = 'm', rsuffix='_m')
    df_hist.loc[:,'z'] = (df_hist.s_m - df_hist.Es) / np.sqrt(df_hist.Vs)
    #df_hist.loc[:,'pval'] = df_hist.z.apply(lambda z : norm.cdf(-np.abs(z)))
    df_hist.loc[:,'pval'] = df_hist.z.apply(lambda z : norm.sf(z))

    # handle the case m=1 seperately
    n1 = df_hist[(df_hist.n1 == 1) & (df_hist.n2 == 0)].N.values
    n2 = df_hist[(df_hist.n1 == 0) & (df_hist.n2 == 1)].N.values
    if len(n1) + len(n2) >= 2 :
        df_hist.loc[df_hist.m == 1, 'pval'] = binom_test_two_sided(n1, n1 + n2 , 1/2)[0]

    return df_hist

def binom_var_test(c1, c2, sym=False, max_m=-1) :
    """ Binmial variance test along stripes
    Args:
    ----
    c1, c2 : list of integers represents count data from two sample
    sym : flag indicates wether the size of both sample is assumed
          identical, hence p=1/2
    """
    df_hist = binom_var_test_df(c1, c2, sym=sym, max_m=max_m)
    return df_hist.groupby('m').pval.mean()

def two_sample_pvals(c1, c2, randomize=False,
     sym=False, alt='two-sided', ret_p=False):

    """ feature by feature exact binomial test
    Args:
    ----
    c1, c2 : list of integers represents count data from two sample
    randomize : flag indicate wether to use randomized P-values
    sym : flag indicates wether the size of both sample is assumed
          identical, hence p=1/2
    alt :  how to compute P-values. 
    """

    T1 = c1.sum()
    T2 = c2.sum()

    den = (T1 + T2 - c1 - c2)
    if den.sum() == 0 :
        return c1 * np.nan

    p = ((T1 - c1) / den)*(1-sym) + sym * 1./2

    if alt == 'greater' or alt == 'less' :
        pvals = binom_test(c1, c1 + c2, p, alt=alt)
    elif randomize :
        pvals = binom_test_two_sided_random(c1, c1 + c2, p)
    else :
        pvals = binom_test_two_sided(c1, c1 + c2, p)

    if ret_p :
        return pvals, p
    return pvals

def two_sample_test_df(X, Y, gamma=0.25, min_cnt=0,
                stbl=True, randomize=False, 
                alt='two-sided', HCtype='HCstar'):
    """
    Same as two_sample_test but returns all information for computing
    HC score of the two samples as a pandas data DataFrame. 
    Requires pandas.

    Args: 
    -----
    X, Y       lists of integers of equal length
    gamma      parameter of HC statistic
    stbl       parameter of HC statistic
    randomize  use randomized or not exact binomial test
    alt        type of test alternatives: 'two-sided' or 'one-sided'
    HCtype     either 'HCstar' (default) or  'original'. Determine
               different variations of HC statistic 

    Returns:
    -------
    counts : DataFrame with fields: 
            n1, n2, p, T1, T2, pval, sign, HC, thresh
            Here: 
            -----
            'n1' <- X
            'n2' <- Y
            'T1' <- sum(X)
            'T2' <- sum(Y)
            'p' <- (T1 - n1) / (T1+ T2 - n1 - n2)
            'pval' <- binom_test(n1, n1 + n2, p) (P-value of test)
            'sign' :    indicates whether a feature is more frequent 
                    in sample X (+1) or sample Y (-1)
            'HC' :      is the higher criticism statistic applies to the
                    column 'pval'
            'thresh' :  indicates whether a feature is below the HC 
                        threshold (True) or not (False)
    """
    import pandas as pd

    counts = pd.DataFrame()
    counts['n1'] = X
    counts['n2'] = Y
    T1 = counts['n1'].sum()
    T2 = counts['n2'].sum()
    
    counts['T1'] = T1
    counts['T2'] = T2

    counts['pval'], counts['p'] = two_sample_pvals(
        counts['n1'], counts['n2'],
        randomize=randomize, alt=alt, ret_p=True)

    counts['sign'] = np.sign(counts.n1 - (counts.n1 + counts.n2) * counts.p)

    counts.loc[counts.n1 + counts.n2 < min_cnt, 'pval'] = np.nan
    pvals = counts.pval.values
    hc = HC(pvals[~np.isnan(pvals)], stbl=stbl)
    #hc_star, p_val_thresh = hc_vals(pvals, gamma=gamma, stbl=stbl)
    if HCtype == 'original' :
        hc, p_val_thresh = hc.HC(gamma=gamma)
    else :
        hc, p_val_thresh = hc.HCstar(gamma=gamma)

    counts['HC'] = hc

    counts['thresh'] = True
    counts.loc[counts['pval'] >= p_val_thresh, ('thresh')] = False
    counts.loc[np.isnan(counts['pval']), ('thresh')] = False
    
    return counts
