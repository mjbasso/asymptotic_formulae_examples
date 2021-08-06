#!/usr/bin/env python3
import itertools
import logging
import time

import numpy as np
from iminuit import Minuit
from probfit import rename
from probfit import SimultaneousFit
from probfit import UnbinnedLH
from scipy.misc import derivative
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.special import ndtri
from scipy.special import xlogy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s'))
logger.addHandler(sh)

_REPLACE_ZEROS_WITH = 1e-3

def poisson(x, mu):
    '''Return the value of a Poisson PDF (with mean mu) at x'''
    mu     = np.maximum(mu, 0.) # Require mu to be positive
    retval = np.exp(xlogy(x, mu) - mu - gammaln(x + 1))
    return np.maximum(retval, 0.)

def ptoys(ntoys, *args):
    '''Return toys drawn from a Poisson PDF (ntoys) with the means for each PDF given by args'''
    for i in range(ntoys):
        yield np.array([np.random.poisson(mu, 1) for mu in args])

def Z0(t0, t1, delete_nan = False):
    '''Calculate the asymtotic significance of discovery given t0, t1 test statistic distributions'''
    t0, t1 = np.array(t0), np.array(t1)
    nan_t0, nan_t1 = list(np.argwhere(np.isnan(t0)).flatten()), list(np.argwhere(np.isnan(t1)).flatten())
    to_delete = list(set(nan_t0 + nan_t1))
    if len(to_delete) != 0:
        if delete_nan:
            # Assumes 1-to-1 correspondence between element i of t0 and element i of t1
            # So we delete both elements if a nan is encountered in either element
            logger.warning('Ignoring %s events due to nan\'s.' % len(to_delete))
            t0 = np.delete(t0, to_delete, axis = 0)
            t1 = np.delete(t1, to_delete, axis = 0)
        else:
            raise RuntimeError('%s nan\'s encountered in t0 and %s nan\'s encountered in t1!' % (len(nan_t0), len(nan_t1)))
    p  = np.sum(t0 > np.median(t1)) / len(t0)
    Z0 = ndtri(1 - p)
    return Z0

def test_statistic(numer_t, denom_t):
    '''Calculate our test statistic, given the likelihoods for the numerator and the denominator'''
    if numer_t == 0. and denom_t == 0.:
        return 0.
    elif numer_t == 0.:
        return np.PINF
    elif denom_t == 0.:
        return np.NINF
    else:
        return -2. * (np.log(numer_t) - np.log(denom_t))

def nCRZ0(s, b, tau):
    ''' Calculate the asymptotic significance for a 1 SR + N CRs measurement
        s   := expected signal yield in SR (float)
        b   := expected background yields in SR (vector of floats, size N)
        tau := transfer coefficients, tau[i][j] carries background i yield in SR to CR j (matrix of floats, shape NxN)
        Returns Z0 (float) '''

    # Argument checking
    b, tau = np.array(b), np.array(tau)
    s, b, tau = float(s), b.astype(float), tau.astype(float)
    assert b.ndim == 1 # b should be a vector
    assert tau.ndim == 2 # tau should be a matrix
    assert len(b) == tau.shape[0] == tau.shape[1]
    assert (tau >= 0.).all() # Assert tau is a transfer matrix
    n = s + np.sum(b)

    # System of equations
    def func(bhh):
        eqns = []
        for k in range(len(b)):
            eqns.append(n / np.sum(bhh) - 1. + np.sum([tau[i, k] * (np.matmul(tau, b)[i] / np.matmul(tau, bhh)[i] - 1.) for i in range(len(b))]))
        return eqns

    # Perform our minimization
    res = least_squares(func, x0 = b, bounds = [tuple(len(b) * [0.]), tuple(len(b) * [np.inf])])
    if not res.success:
        raise RuntimeError('Minimization failed: status = %s, message = \'%s\'' % (res.status, res.message))
    bhh = np.array(res.x)

    # Calculate our significance
    Z0 = np.sqrt(-2. * (n * np.log(np.sum(bhh) / n) + n
        + np.sum([-1. * bhh[i] + np.matmul(tau, b)[i] * np.log(np.matmul(tau, bhh)[i] / np.matmul(tau, b)[i]) + np.matmul(tau, b - bhh)[i] for i in range(len(b))])))
    return Z0

def nCRZ0_MC(s, b, tau, ntoys = 50000, return_t0_and_t1 = False, sleep = 0.002, throw_toys_only = False, max_retries = 3, skip_failed_toys = False, retry_first = True):
    ''' Calculate the significance for a 1 SR + N CRs measurement using toys
        s                := expected signal yield in SR (float)
        b                := expected background yields in SR (vector of floats, size N)
        tau              := transfer coefficients, tau[i][j] carries background i yield in SR to CR j (matrix of floats, shape NxN)
        return_t0_and_t1 := return a tuple of Z0 (float), t0 (vector of floats), and t1 (vector of floats)
        sleep            := time to sleep after each toy fit (to prevent 100% CPU usage) (float)
        throw_toys_only  := only throw toys, don't perform fits or calculate Z0 -- for debugging (bool)
        max_retries      := number of times to retry the entire loop (if a fit fails) (int)
        skip_failed_toys := discard toys with failed minimization (**could introduce bias**) (bool)
        retry_first      := retry the entire loop first *prior* to discarding toys (bool)
        Returns Z0 (float) '''

    # Argument checking
    b, tau = np.array(b), np.array(tau)
    s, b, tau = float(s), b.astype(float), tau.astype(float)
    assert b.ndim == 1 # b should be a vector
    assert tau.ndim == 2 # tau should be a matrix
    assert len(b) == tau.shape[0] == tau.shape[1]
    assert (tau >= 0.).all() # Assert tau is a transfer matrix
    assert max_retries >= 0

    # Create our unconditional (s+b), conditional (b only), and CR PDFs
    args  = ['b%s' % i for i in range(len(b))]
    _args = ['_' + arg for arg in args]
    kwargs = {args[j]: b[j] for j in range(len(b))}
    kwargs.update(dict(errordef = Minuit.LIKELIHOOD))
    pdf_SR_uncond = rename(eval('lambda _x, _s, %s: poisson(_x, _s + %s)' % (', '.join(_args), ' + '.join(_args)), {'poisson': poisson}), tuple(['x', 's'] + args))
    pdf_SR_cond   = rename(eval('lambda _x, %s: poisson(_x, %s)' % (', '.join(_args), ' + '.join(_args)), {'poisson': poisson}), tuple(['x'] + args))
    pdf_CRs       = [rename(eval('lambda _x, %s: poisson(_x, matmul(tau, [%s])[%s])' % (', '.join(_args), ', '.join(_args), i),
        {'poisson': poisson, 'matmul': np.matmul, 'tau': tau}), tuple(['x'] + args)) for i in range(len(b))]

    # t0, t1 are our H0, H1 simulated values of the test statistic
    t0 = []
    t1 = []

    retry = 0
    nskipped = 0
    while True:
        try:

            # Iterate over our toys for both H0 (t0) and H1 (t1)
            for i, (n_t0, m_t0, n_t1, m_t1) in enumerate(zip(ptoys(ntoys, np.sum(b)), ptoys(ntoys, *[np.matmul(tau, b)[j] for j in range(len(b))]),
                ptoys(ntoys, s + np.sum(b)), ptoys(ntoys, *[np.matmul(tau, b)[j] for j in range(len(b))]))):

                if i % 1000 == 0:
                    logger.info('On %s/%s.' % (i, ntoys))

                if throw_toys_only:
                    time.sleep(sleep / 100.)
                    continue

                n_t0, n_t1 = n_t0[0], n_t1[0]

                logger.debug('Loop %s, inputs: s = %s, b = %s, tau = %s' % (i, s, b.tolist(), tau.tolist()))
                logger.debug('Loop %s, toys: n_t0 = %s, m_t0 = %s, n_t1 = %s, m_t1 = %s' % (i, n_t0[0], m_t0.tolist(), n_t1[0], m_t1.tolist()))

                n_t0, n_t1, m_t0, m_t1 = n_t0.astype(float), n_t1.astype(float), m_t0.astype(float), m_t1.astype(float)

                replaced_zeros = False
                if not n_t0[0]:    n_t0[0]         = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not n_t1[0]:    n_t1[0]         = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not m_t0.all(): m_t0[m_t0 == 0] = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not m_t1.all(): m_t1[m_t1 == 0] = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if replaced_zeros:
                    logger.warning('Loop %s, replaced 0\'s in thrown yields with %s.' % (i, _REPLACE_ZEROS_WITH))

                # Set up our LHs (unconditional+condtional) for H0 (unbinned LHs in all regions)
                simlh_uncond_t0 = SimultaneousFit(UnbinnedLH(pdf_SR_uncond, n_t0), *[UnbinnedLH(pdf_CRs[j], m_t0[j]) for j in range(len(b))])
                simlh_cond_t0   = SimultaneousFit(UnbinnedLH(pdf_SR_cond,   n_t0), *[UnbinnedLH(pdf_CRs[j], m_t0[j]) for j in range(len(b))])

                # Minimize both
                m_uncond_t0 = Minuit(simlh_uncond_t0, s = s, **kwargs)
                m_uncond_t0.migrad()
                logger.debug('Loop %s, m_uncond_t0 result: %s' % (i, dict(m_uncond_t0.values)))
                if not (m_uncond_t0.valid and m_uncond_t0.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s: minimization failed!' % i)
                m_cond_t0 = Minuit(simlh_cond_t0, **kwargs)
                m_cond_t0.migrad()
                logger.debug('Loop %s, m_cond_t0 result: %s' % (i, dict(m_cond_t0.values)))
                if not (m_cond_t0.valid and m_cond_t0.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s: minimization failed!' % i)

                # If our measured value of s is less than 0, set t0 to 0
                if m_uncond_t0.values['s'] < 0:
                    t0.append(0.)
                # Else, evaluate our LHs in the numerator and denominator of t0 and actually calculate it
                else:
                    b_cond   = [m_cond_t0.values['b%s' % j] for j in range(len(b))]
                    s_uncond =  m_uncond_t0.values['s']
                    b_uncond = [m_uncond_t0.values['b%s' % j] for j in range(len(b))]
                    numer_t0 = pdf_SR_cond(n_t0, *b_cond) * np.prod([pdf_CRs[j](m_t0[j], *b_cond) for j in range(len(b))])
                    denom_t0 = pdf_SR_uncond(n_t0, s_uncond, *b_uncond) * np.prod([pdf_CRs[j](m_t0[j], *b_uncond) for j in range(len(b))])
                    t0.append(test_statistic(numer_t0[0], denom_t0[0]))

                # Set up our LHs (unconditional+condtional) for H1 (unbinned LHs in all regions)
                simlh_uncond_t1 = SimultaneousFit(UnbinnedLH(pdf_SR_uncond, n_t1), *[UnbinnedLH(pdf_CRs[j], m_t1[j]) for j in range(len(b))])
                simlh_cond_t1   = SimultaneousFit(UnbinnedLH(pdf_SR_cond,   n_t1), *[UnbinnedLH(pdf_CRs[j], m_t1[j]) for j in range(len(b))])

                # Minimize both
                m_uncond_t1 = Minuit(simlh_uncond_t1, s = s, **kwargs)
                m_uncond_t1.migrad()
                logger.debug('Loop %s, m_uncond_t1 result: %s' % (i, dict(m_uncond_t1.values)))
                if not (m_uncond_t1.valid and m_uncond_t1.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s: minimization failed!' % i)
                m_cond_t1 = Minuit(simlh_cond_t1, **kwargs)
                m_cond_t1.migrad()
                logger.debug('Loop %s, m_cond_t1 result: %s' % (i, dict(m_cond_t1.values)))
                if not (m_cond_t1.valid and m_cond_t1.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s: minimization failed!' % i)

                # If our measured value of s is less than 0, set t1 to 0
                if m_uncond_t1.values['s'] < 0:
                    t1.append(0.)
                # Else, evaluate our LHs in the numerator and denominator of t1 and actually calculate it
                else:
                    b_cond   = [m_cond_t1.values['b%s' % j] for j in range(len(b))]
                    s_uncond =  m_uncond_t1.values['s']
                    b_uncond = [m_uncond_t1.values['b%s' % j] for j in range(len(b))]
                    numer_t1 = pdf_SR_cond(n_t1, *b_cond) * np.prod([pdf_CRs[j](m_t1[j], *b_cond) for j in range(len(b))])
                    denom_t1 = pdf_SR_uncond(n_t1, s_uncond, *b_uncond) * np.prod([pdf_CRs[j](m_t1[j], *b_uncond) for j in range(len(b))])
                    t1.append(test_statistic(numer_t1[0], denom_t1[0]))

                time.sleep(sleep)

            break

        except RuntimeError as e:
            if retry < max_retries:
                logger.warning('Exception raised (\"%s\"), retrying full loop (%s/%s)!' % (e, retry, max_retries))
                retry += 1
                continue
            else:
                logger.error('Exception raised (\"%s\") and max number of retries reached (%s/%s), exiting!' % (e, retry, max_retries))
                raise RuntimeError(e)
    pass # while True

    if throw_toys_only:
        if return_t0_and_t1:
            return (None, None, None)
        else:
            return None

    z0 = Z0(t0, t1)
    logger.info('Calculated Z0: %s' % z0)
    if nskipped > 0:
        logger.warning('%s toys skipped, results may be biased.' % nskipped)
    if return_t0_and_t1:
        return (z0, t0, t1)
    return z0

def nSRZ0(s, b, tau):
    ''' Calculate the asymptotic significance for a N SRs + 1 CR measurement
        s   := expected signal yields in SRs (vector of floats, size N)
        b   := expected background yield in CR (float)
        tau := transfer coefficients, 1/tau[i] carries background yield in CR to SR i (vector of floats, size N)
        Returns Z0 (float) '''

    # Argument checking
    s, tau = np.array(s), np.array(tau)
    s, b, tau = s.astype(float), float(b), tau.astype(float)
    assert s.ndim == 1 # s should be a vector
    assert tau.ndim == 1 # tau should be a vector
    assert len(s) == len(tau)
    n = s + b / tau
    bhh = (b + np.sum(n)) / (1. + np.sum(1. / tau))

    # Calculate our significance
    Z0 = np.sqrt(-2. * (np.sum([n[i] * np.log(bhh / tau[i] / n[i]) + n[i] - bhh / tau[i] for i in range(len(n))]) + b * np.log(bhh / b) + (b - bhh)))
    return Z0

def nSRZ0_MC(s, b, tau, ntoys = 50000, return_t0_and_t1 = False, sleep = 0.002, throw_toys_only = False, max_retries = 3, skip_failed_toys = False, retry_first = True):
    ''' Calculate the asymptotic significance for a N SRs + 1 CR measurement
        s                := expected signal yields in SRs (vector of floats, size N)
        b                := expected background yield in CR (float)
        tau              := transfer coefficients, 1/tau[i] carries background yield in CR to SR i (vector of floats, size N)
        return_t0_and_t1 := return a tuple of Z0 (float), t0 (vector of floats), and t1 (vector of floats)
        sleep            := time to sleep after each toy fit (to prevent 100% CPU usage) (float)
        throw_toys_only  := only throw toys, don't perform fits or calculate Z0 -- for debugging (bool)
        max_retries      := number of times to retry the entire loop (if a fit fails) (int)
        skip_failed_toys := discard toys with failed minimization (**could introduce bias**) (bool)
        retry_first      := retry the entire loop first *prior* to discarding toys (bool)
        Returns Z0 (float) '''

    # Argument checking
    s, tau = np.array(s), np.array(tau)
    s, b, tau = s.astype(float), float(b), tau.astype(float)
    assert s.ndim == 1 # s should be a vector
    assert tau.ndim == 1 # tau should be a vector
    assert len(s) == len(tau)
    assert max_retries >= 0

    # Create our unconditional (s+b), conditional (b only), and CR PDFs
    pdf_SRs_uncond = [rename(eval('lambda _x, _mu, _b: poisson(_x, _mu * s[%s] + _b / tau[%s])' % (i, i), {'poisson': poisson, 's': s, 'tau': tau}), ('x', 'mu', 'b')) for i in range(len(s))]
    pdf_SRs_cond   = [rename(eval('lambda _x, _b: poisson(_x, _b / tau[%s])' % i, {'poisson': poisson, 'tau': tau}), ('x', 'b')) for i in range(len(s))]
    pdf_CR         = rename(lambda _x, _b: poisson(_x, _b), ('x', 'b'))

    # t0, t1 are our H0, H1 simulated values of the test statistic
    t0 = []
    t1 = []

    retry = 0
    nskipped = 0
    while True:
        try:

            # Iterate over our toys for both H0 (t0) and H1 (t1)
            for i, (n_t0, m_t0, n_t1, m_t1) in enumerate(zip(ptoys(ntoys, *(b / tau)), ptoys(ntoys, b), ptoys(ntoys, *(s + b / tau)), ptoys(ntoys, b))):

                if i % 1000 == 0:
                    logger.info('On %s/%s.' % (i, ntoys))

                if throw_toys_only:
                    time.sleep(sleep / 100.)
                    continue

                m_t0, m_t1 = m_t0[0], m_t1[0]

                logger.debug('Loop %s, inputs: s = %s, b = %s, tau = %s' % (i, s.tolist(), b, tau.tolist()))
                logger.debug('Loop %s, toys: n_t0 = %s, m_t0 = %s, n_t1 = %s, m_t1 = %s' % (i, n_t0.tolist(), m_t0[0], n_t1.tolist(), m_t1[0]))

                n_t0, n_t1, m_t0, m_t1 = n_t0.astype(float), n_t1.astype(float), m_t0.astype(float), m_t1.astype(float)

                replaced_zeros = False
                if not n_t0.all(): n_t0[n_t0 == 0] = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not n_t1.all(): n_t1[n_t0 == 0] = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not m_t0[0]:    m_t0[0]         = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not m_t1[0]:    m_t1[0]         = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if replaced_zeros:
                    logger.warning('Loop %s, replaced 0\'s in thrown yields with %s.' % (i, _REPLACE_ZEROS_WITH))

                # Set up our LHs (unconditional+condtional) for H0 (unbinned LHs in all regions)
                simlh_uncond_t0 = SimultaneousFit(UnbinnedLH(pdf_CR, m_t0), *[UnbinnedLH(pdf_SRs_uncond[j], n_t0[j]) for j in range(len(s))])
                simlh_cond_t0   = SimultaneousFit(UnbinnedLH(pdf_CR, m_t0), *[UnbinnedLH(pdf_SRs_cond[j],   n_t0[j]) for j in range(len(s))])

                # Minimize both
                m_uncond_t0 = Minuit(simlh_uncond_t0, errordef = Minuit.LIKELIHOOD, mu = 1., b = b)
                m_uncond_t0.migrad()
                logger.debug('Loop %s, m_uncond_t0 result: %s' % (i, dict(m_uncond_t0.values)))
                if not (m_uncond_t0.valid and m_uncond_t0.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)
                m_cond_t0 = Minuit(simlh_cond_t0, errordef = Minuit.LIKELIHOOD, b = b)
                m_cond_t0.migrad()
                logger.debug('Loop %s, m_cond_t0 result: %s' % (i, dict(m_cond_t0.values)))
                if not (m_cond_t0.valid and m_cond_t0.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)

                # If our measured value of mu is less than 0, set t0 to 0
                if m_uncond_t0.values['mu'] < 0:
                    t0.append(0.)
                # Else, evaluate our LHs in the numerator and denominator of t0 and actually calculate it
                else:
                    b_cond    = m_cond_t0.values['b']
                    mu_uncond = m_uncond_t0.values['mu']
                    b_uncond  = m_uncond_t0.values['b']
                    numer_t0  = np.prod([pdf_SRs_cond[j](n_t0[j], b_cond) for j in range(len(s))]) * pdf_CR(m_t0, b_cond)
                    denom_t0  = np.prod([pdf_SRs_uncond[j](n_t0[j], mu_uncond, b_uncond) for j in range(len(s))]) * pdf_CR(m_t0, b_uncond)
                    t0.append(test_statistic(numer_t0[0], denom_t0[0]))

                # Set up our LHs (unconditional+condtional) for H1 (unbinned LHs in all regions)
                simlh_uncond_t1 = SimultaneousFit(UnbinnedLH(pdf_CR, m_t1), *[UnbinnedLH(pdf_SRs_uncond[j], n_t1[j]) for j in range(len(s))])
                simlh_cond_t1   = SimultaneousFit(UnbinnedLH(pdf_CR, m_t1), *[UnbinnedLH(pdf_SRs_cond[j],   n_t1[j]) for j in range(len(s))])

                # Minimize both
                m_uncond_t1 = Minuit(simlh_uncond_t1, errordef = Minuit.LIKELIHOOD, mu = 1., b = b)
                m_uncond_t1.migrad()
                logger.debug('Loop %s, m_uncond_t1 result: %s' % (i, dict(m_uncond_t1.values)))
                if not (m_uncond_t1.valid and m_uncond_t1.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)
                m_cond_t1 = Minuit(simlh_cond_t1, errordef = Minuit.LIKELIHOOD, b = b)
                m_cond_t1.migrad()
                logger.debug('Loop %s, m_cond_t1 result: %s' % (i, dict(m_cond_t1.values)))
                if not (m_cond_t1.valid and m_cond_t1.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)

                # If our measured value of mu is less than 0, set t1 to 0
                if m_uncond_t1.values['mu'] < 0:
                    t1.append(0.)
                # Else, evaluate our LHs in the numerator and denominator of t1 and actually calculate it
                else:
                    b_cond    = m_cond_t1.values['b']
                    mu_uncond = m_uncond_t1.values['mu']
                    b_uncond  = m_uncond_t1.values['b']
                    numer_t1  = np.prod([pdf_SRs_cond[j](n_t1[j], b_cond) for j in range(len(s))]) * pdf_CR(m_t1, b_cond)
                    denom_t1  = np.prod([pdf_SRs_uncond[j](n_t1[j], mu_uncond, b_uncond) for j in range(len(s))]) * pdf_CR(m_t1, b_uncond)
                    t1.append(test_statistic(numer_t1[0], denom_t1[0]))

                time.sleep(sleep)

            break

        except RuntimeError as e:
            if retry < max_retries:
                logger.warning('Exception raised (\"%s\"), retrying full loop (%s/%s)!' % (e, retry, max_retries))
                retry += 1
                continue
            else:
                logger.error('Exception raised (\"%s\") and max number of retries reached (%s/%s), exiting!' % (e, retry, max_retries))
                raise RuntimeError(e)
    pass # while True

    if throw_toys_only:
        if return_t0_and_t1:
            return (None, None, None)
        else:
            return None

    z0 = Z0(t0, t1)
    logger.info('Calculated Z0: %s' % z0)
    if nskipped > 0:
        logger.warning('%s toys skipped, results may be biased.' % nskipped)
    if return_t0_and_t1:
        return (z0, t0, t1)
    return z0

def GaussZ0(s, b, R, S, abstol = 1e-6):
    ''' Calculate the asymptotic significance for a 1 SR + N backgrounds + M Gaussian constraints measurement
        s      := expected signal yield in SR (float)
        b      := expected background yield in CR (float)
        R      := response matrix, R[i][j] is a function which describes the response (i.e., multiplies the yield) of background i as function of nuisance parameter j
                  (matrix of functions, shape NxM)
        S      := correlation matrix, S[i][j] is a float between -1 and +1 which describes the correlation between nuisance parameters i and j (matrix of floats, shape MxM)
        abstol := absolute tolerance (within 0) required for the minimization to succeed
        Returns Z0 (float) '''

    # Argument checking
    b, R, S = np.array(b), np.array(R), np.array(S)
    s, b, S = float(s), b.astype(float), S.astype(float)
    assert b.ndim == 1 # b should be a vector
    assert R.ndim == 2 # R should be a matrix
    assert S.ndim == 2 # S should be a matrix
    assert (S.transpose() == S).all() and (S <= +1.).all() and (S >= -1.).all() # Assert S is a correlation matrix
    assert R.shape == (len(b), S.shape[0])
    n = s + np.sum(b)
    Si = np.linalg.inv(S)

    # System of equations
    def func(th):
        eqns = []
        RdotB = []
        prefactor = 0.
        for i in range(len(b)):
            RdotB.append(np.prod([R[i, j](th[j]) for j in range(S.shape[0])]) * b[i])
            prefactor += RdotB[i]
        prefactor = n / prefactor - 1.
        for j in range(S.shape[0]):
            term1 = 0.
            for i in range(len(b)):
                term1 += derivative(R[i, j], th[j], dx = 1e-6) * RdotB[i] / R[i, j](th[j])
            eqns.append(prefactor * term1 - np.sum(Si[j, :] * th))
        return eqns

    # Perform our minimization
    x0 = [np.zeros(S.shape[0])] + [np.array(prod) for prod in itertools.product([+1., -1.], repeat = S.shape[0])]
    success = False
    for _x0 in x0:
        logger.debug('Running with initial guess: x0 = %s' % _x0)
        res = least_squares(func, x0 = _x0, bounds = [tuple(S.shape[0] * [-3.]), tuple(S.shape[0] * [+3.])]) # Note: the bounds might have to be changed to suit the problem
        if not res.success:
            logger.debug('Minimization failed (status = %s, message = \'%s\'), re-running with new initial guess.' % (res.status, res.message))
            continue
        th, fun = np.array(res.x), np.array(res.fun)
        if not (np.abs(fun - np.zeros(S.shape[0])) < abstol).all():
            logger.debug('Minimization failed to achieve absolute tolerance (results = %s, abstol = %s), re-running with new initial guess.' % (fun, abstol))
            continue
        else:
            success = True
            break
    if not success:
        raise RuntimeError('Minimization failed to achieve absolute tolerance (abstol = %s) for all initial guesses!' % abstol)
    sumRdotB = 0.
    for i in range(len(b)):
        sumRdotB += np.prod([R[i, j](th[j]) for j in range(len(th))]) * b[i]

    # Calculate our significance
    Z0 = np.sqrt(-2. * (n * np.log(sumRdotB / n) + n - sumRdotB - 0.5 * np.matmul(np.matmul(th, Si), th.T)))
    return Z0

def gaussian(x, mu, Si, norm):
    '''Return the value of a multivariate Gaussian PDF (with central value mu, inverted correlation matrix Si i.e. Si := S^-1, and normalization factor norm) at x'''
    diff   = x - mu
    retval = np.exp(-0.5 * np.matmul(diff, np.matmul(Si, diff))) / norm
    return np.maximum(retval, 0.)

def apply_response(x, R, th):
    ''' Apply a response matrix R (matrix of functions, shape NxM), evaluated for nuisance parameter values th (vector of floats, size M),
        to input (vector of floats, size N)'''
    y = np.array(x)
    for i in range(len(x)):
        for j in range(len(th)):
            y[i] *= R[i, j](th[j])
    return y

def gtoys(ntoys, *args):
    ''' Return toys drawn from a multivariate Gaussian PDF (ntoys) with the central values/correlation matrices for each PDF given by args
        (i.e., each element in args is a tuple with element 0 the vectorial central value and element 1 the correlation matrix) '''
    for i in range(ntoys):
        yield [np.random.multivariate_normal(arg[0], arg[1], 1) for arg in args]

def GaussZ0_MC(s, b, R, S, ntoys = 50000, return_t0_and_t1 = False, sleep = 0.002, throw_toys_only = False, max_retries = 3, skip_failed_toys = False, retry_first = True):
    ''' Calculate the asymptotic significance for a 1 SR + N backgrounds + M Gaussian constraints measurement
        s                := expected signal yield in SR (float)
        b                := expected background yield in CR (float)
        R                := response matrix, R[i][j] is a function which describes the response (i.e., multiplies the yield) of background i as function of nuisance parameter j
                            (matrix of functions, shape NxM)
        S                := correlation matrix, S[i][j] is a float between -1 and +1 which describes the correlation between nuisance parameters i and j (matrix of floats, shape MxM)
        return_t0_and_t1 := return a tuple of Z0 (float), t0 (vector of floats), and t1 (vector of floats)
        sleep            := time to sleep after each toy fit (to prevent 100% CPU usage) (float)
        throw_toys_only  := only throw toys, don't perform fits or calculate Z0 -- for debugging (bool)
        max_retries      := number of times to retry the entire loop (if a fit fails) (int)
        skip_failed_toys := discard toys with failed minimization (**could introduce bias**) (bool)
        retry_first      := retry the entire loop first *prior* to discarding toys (bool)
        Returns Z0 (float) '''

    # Argument checking
    b, R, S = np.array(b), np.array(R), np.array(S)
    s, b, S = float(s), b.astype(float), S.astype(float)
    assert b.ndim == 1 # b should be a vector
    assert R.ndim == 2 # R should be a matrix
    assert S.ndim == 2 # S should be a matrix
    assert (S.transpose() == S).all() and (S <= +1.).all() and (S >= -1.).all() # Assert S is a correlation matrix
    assert R.shape == (len(b), len(S))
    assert max_retries >= 0

    Si = np.linalg.inv(S)
    norm = np.sqrt(np.abs(np.linalg.det(S)) * (2 * np.pi) ** len(S))

    args  = ['th%s' % i for i in range(len(S))]
    _args = ['_' + arg for arg in args]

    # t0, t1 are our H0, H1 simulated values of the test statistic
    t0 = []
    t1 = []

    # Handy function for testing different fits
    def fit(lh, args, th_central, s = None, offset = 5.):
        base_multipliers     = [0., +1., -1.]
        detailed_multipliers = base_multipliers + [+4., +2., +0.5, +0.25, -0.25, -0.5, -2., -4.]
        tried_detailed       = False
        multipliers          = base_multipliers
        tried_multipliers    = set()
        while True:
            for multiplier in itertools.product(multipliers, repeat = len(th_central)):
                if multiplier in tried_multipliers:
                    continue
                tried_multipliers.add(multiplier)
                _th_central = th_central + np.array(multiplier) * offset
                logger.debug('Running with initial guess: x0 = %s' % _th_central)
                kwargs = {args[i]: _th_central[i] for i in range(len(th_central))}
                kwargs['errordef'] = Minuit.LIKELIHOOD
                if s is not None: kwargs['s'] = s
                minu = Minuit(lh, **kwargs)
                minu.migrad()
                if not (minu.valid and minu.accurate):
                    logger.debug('Minimization failed, re-running with new initial guess.')
                    continue
                break
            if not (minu.valid and minu.accurate) and not tried_detailed:
                multipliers    = detailed_multipliers
                tried_detailed = True
                continue
            break
        return minu

    retry = 0
    nskipped = 0
    while True:
        try:

            # Iterate over our toys for both H0 (t0) and H1 (t1)
            for i, (n_t0, n_t1, th_central) in enumerate(zip(ptoys(ntoys, np.sum(b)), ptoys(ntoys, s + np.sum(b)), gtoys(ntoys, (np.zeros(len(S)), S)))):

                if i % 1000 == 0:
                    logger.info('On %s/%s.' % (i, ntoys))

                if throw_toys_only:
                    time.sleep(sleep / 100.)
                    continue

                n_t0, n_t1, th_central = n_t0[0], n_t1[0], th_central[0][0]

                logger.debug('Loop %s, inputs: s = %s, b = %s, R = %s, S = %s' % (i, s, b.tolist(), R.tolist(), S.tolist()))
                logger.debug('Loop %s, toys: n_t0 = %s, n_t1 = %s, th_central = %s' % (i, n_t0[0], n_t1[0], th_central.tolist()))

                n_t0, n_t1, th_central = n_t0.astype(float), n_t1.astype(float), th_central.astype(float)

                replaced_zeros = False
                if not n_t0[0]: n_t0[0] = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if not n_t1[0]: n_t1[0] = _REPLACE_ZEROS_WITH; replaced_zeros = True
                if replaced_zeros:
                    logger.warning('Loop %s, replaced 0\'s in thrown yields with %s.' % (i, _REPLACE_ZEROS_WITH))

                # Create our unconditional (s+b) and conditional (b only) PDFs using our sample
                pdf_SR_uncond = rename(eval('lambda _x, _s, %s: poisson(_x, _s + sum(apply_response(b, R, [%s]))) * gaussian([%s], th_central, Si, norm)' % tuple(3 * [', '.join(_args)]),
                    {'poisson': poisson, 'sum': np.sum, 'apply_response': apply_response, 'b': b, 'R': R, 'gaussian': gaussian, 'th_central': th_central, 'Si': Si, 'norm': norm}),
                    tuple(['x', 's'] + args))
                pdf_SR_cond = rename(eval('lambda _x, %s: poisson(_x, sum(apply_response(b, R, [%s]))) * gaussian([%s], th_central, Si, norm)' % tuple(3 * [', '.join(_args)]),
                    {'poisson': poisson, 'sum': np.sum, 'apply_response': apply_response, 'b': b, 'R': R, 'gaussian': gaussian, 'th_central': th_central, 'Si': Si, 'norm': norm}),
                    tuple(['x'] + args))

                # Set up our LHs (unconditional+condtional) for H0 (unbinned LHs in all regions)
                simlh_uncond_t0 = UnbinnedLH(pdf_SR_uncond, n_t0)
                simlh_cond_t0   = UnbinnedLH(pdf_SR_cond,   n_t0)

                # Minimize both
                # Very empirical offset:
                # --> The offset describes the step size along the different directions of th_central for finding a valid solution
                base_response = abs(apply_response(b, R, th_central))
                base_response = np.ones(len(base_response)) if np.all(base_response == 0.) else base_response
                base_response /= np.linalg.norm(base_response)
                real_response = np.sum(apply_response(b, R, th_central))
                offset = 5. if (n_t0 > _REPLACE_ZEROS_WITH) else 3. * abs(n_t0[0] / (real_response if real_response != 0. else 1.) - 1.) * base_response
                m_uncond_t0 = fit(simlh_uncond_t0, args, th_central, s, offset = offset)
                logger.debug('Loop %s, m_uncond_t0 result: %s' % (i, dict(m_uncond_t0.values)))
                if not (m_uncond_t0.valid and m_uncond_t0.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)
                m_cond_t0 = fit(simlh_cond_t0, args, th_central, offset = offset)
                logger.debug('Loop %s, m_cond_t0 result: %s' % (i, dict(m_cond_t0.values)))
                if not (m_cond_t0.valid and m_cond_t0.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)

                # If our measured value of s is less than 0, set t0 to 0
                if m_uncond_t0.values['s'] < 0:
                    t0.append(0.)
                # Else, evaluate our LHs in the numerator and denominator of t0 and actually calculate it
                else:
                    numer_t0 = pdf_SR_cond(n_t0, *[m_cond_t0.values[arg] for arg in args])
                    denom_t0 = pdf_SR_uncond(n_t0, m_uncond_t0.values['s'], *[m_uncond_t0.values[arg] for arg in args])
                    t0.append(test_statistic(numer_t0[0], denom_t0[0]))

                # Set up our LHs (unconditional+condtional) for H1 (unbinned LHs in all regions)
                simlh_uncond_t1 = UnbinnedLH(pdf_SR_uncond, n_t1)
                simlh_cond_t1   = UnbinnedLH(pdf_SR_cond,   n_t1)

                # Minimize both
                # Very empirical offset:
                # --> The offset describes the step size along the different directions of th_central for finding a valid solution
                offset = 5. if (n_t1 > _REPLACE_ZEROS_WITH) else 3. * abs(n_t1[0] / ((s + real_response) if (s + real_response) != 0. else 1.) - 1.) * base_response
                m_uncond_t1 = fit(simlh_uncond_t1, args, th_central, s, offset = offset)
                logger.debug('Loop %s, m_uncond_t1 result: %s' % (i, dict(m_uncond_t1.values)))
                if not (m_uncond_t1.valid and m_uncond_t1.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)
                m_cond_t1 = fit(simlh_cond_t1, args, th_central, offset = offset)
                logger.debug('Loop %s, m_cond_t1 result: %s' % (i, dict(m_cond_t1.values)))
                if not (m_cond_t1.valid and m_cond_t1.accurate):
                    if skip_failed_toys and not (retry_first and retry < max_retries):
                        logger.warning('Loop %s: minimization failed, skipping toy (*may bias results*).' % i)
                        nskipped += 1
                        continue
                    raise RuntimeError('Loop %s, minimization failed!' % i)

                # If our measured value of s is less than 0, set t0 to 0
                if m_uncond_t1.values['s'] < 0:
                    t1.append(0.)
                # Else, evaluate our LHs in the numerator and denominator of t0 and actually calculate it
                else:
                    numer_t1 = pdf_SR_cond(n_t1, *[m_cond_t1.values[arg] for arg in args])
                    denom_t1 = pdf_SR_uncond(n_t1, m_uncond_t1.values['s'], *[m_uncond_t1.values[arg] for arg in args])
                    t1.append(test_statistic(numer_t1[0], denom_t1[0]))

                time.sleep(sleep)

            break

        except RuntimeError as e:
            if retry < max_retries:
                logger.warning('Exception raised (\"%s\"), retrying full loop (%s/%s)!' % (e, retry, max_retries))
                retry += 1
                continue
            else:
                logger.error('Exception raised (\"%s\") and max number of retries reached (%s/%s), exiting!' % (e, retry, max_retries))
                raise RuntimeError(e)
    pass # while True

    if throw_toys_only:
        if return_t0_and_t1:
            return (None, None, None)
        else:
            return None

    z0 = Z0(t0, t1)
    logger.info('Calculated Z0: %s' % z0)
    if nskipped > 0:
        logger.warning('%s toys skipped, results may be biased.' % nskipped)
    if return_t0_and_t1:
        return (z0, t0, t1)
    return z0

if __name__ == '__main__':

    run_toys = False

    s = 10.
    b = [8., 5.]
    tau = [[8., 0.], [0., 5.]]
    print(nCRZ0(s, b, tau))
    if run_toys:
        print(nCRZ0_MC(s, b, tau))

    s = [40., 10., 12.]
    b = 1000.
    tau = [2., 10., 20.]
    print(nSRZ0(s, b, tau))
    if run_toys:
        print(nSRZ0_MC(s, b, tau))

    s = 10.
    b = [25., 25.]
    R = [[lambda th: 1. + 0.4 * th, lambda th: 1. + 0.2 * th], [lambda th: 1. + 0.1 * th, lambda th: 1. + 0.1 * th]]
    S = [[1., 0.75], [0.75, 1.]]
    print(GaussZ0(s, b, R, S))
    if run_toys:
        print(GaussZ0_MC(s, b, tau))
