#!/usr/bin/env python3
import logging
import os
import pickle
import time
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import rc
from scipy.optimize import least_squares

import asymptotic_formulae
from asymptotic_formulae import GaussZ0
from asymptotic_formulae import GaussZ0_MC
from asymptotic_formulae import nCRZ0
from asymptotic_formulae import nCRZ0_MC
from asymptotic_formulae import nSRZ0
from asymptotic_formulae import nSRZ0_MC

rc('font', **{'family': 'sans-serif','sans-serif': ['Helvetica']})
rc('text', usetex = True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s'))
logger.addHandler(sh)

# For creating a set of uniformly-spaced points on a log scale
def logVector(low, high, n):
    low  = np.log(low) / np.log(10)
    high = np.log(high) / np.log(10)
    step = (high - low) / n
    vec  = np.array([low + step * i for i in range(n + 1)])
    return np.exp(np.log(10) * vec)

# As described in Section 2.1.4
def nCRZ0_DiagTau(s, b, tau):
    ''' Calculate the asymptotic significance for a 1 SR + N CRs, diagonal tau measurement
        s   := expected signal yield in SR (float)
        b   := expected background yields in SR (vector of floats, size N)
        tau := transfer coefficients, tau[i] carries background i yield in SR to CR i (vector of floats, size N)
        Returns Z0 (float) '''

    # Argument checking
    b, tau = np.array(b), np.array(tau)
    s, b, tau = float(s), b.astype(float), tau.astype(float)
    assert b.ndim == 1 # b should be a vector
    assert tau.ndim == 1 # tau should be a vector
    assert len(b) == len(tau)
    assert (tau >= 0.).all() # Assert tau contains transfer factors (i.e., all positive)
    n = s + np.sum(b)

    # System of equations
    def func(bhh):
        eqns = []
        for k in range(len(b)):
            eqns.append(n / np.sum(bhh) - 1. + tau[k] * (b[k] / bhh[k] - 1.))
        return eqns

    # Perform our minimization
    res = least_squares(func, x0 = b, bounds = [tuple(len(b) * [0.]), tuple(len(b) * [np.inf])])
    if not res.success:
        raise RuntimeError('Minimization failed: status = %s, message = \'%s\'' % (res.status, res.message))
    bhh = np.array(res.x)

    # Calculate our significance
    Z0 = np.sqrt(-2. * np.log((np.sum(bhh) / n) ** n * np.prod([(bhh[k] / b[k]) ** (tau[k] * b[k]) for k in range(len(b))])))
    return Z0

# As described in Section 2.4.2
def GaussZ0_Decorr(s, b, sigma):
    ''' Calculate the asymptotic significance for a 1 SR + N CRs, diagonal tau measurement
        s     := expected signal yield in SR (float)
        b     := expected background yields in SR (vector of floats, size N)
        sigma := width of Gaussian constraint ("absolute uncertainty") for each background yield (vector of floats, size N)
        Returns Z0 (float) '''

    # Argument checking
    b, sigma = np.array(b), np.array(sigma)
    s, b, sigma = float(s), b.astype(float), sigma.astype(float)
    assert b.ndim == 1 # b should be a vector
    assert sigma.ndim == 1 # sigma should be a vector
    assert len(b) == len(sigma)
    assert (sigma >= 0.).all() # Assert sigma contains widths (i.e., all positive)
    n = s + np.sum(b)

    # System of equations
    def func(bhh):
        eqns = []
        for k in range(len(b)):
            eqns.append(sigma[k] * (n / np.sum(bhh) - 1.) - (bhh[k] - b[k]) / sigma[k])
        return eqns

    # Perform our minimization
    res = least_squares(func, x0 = b, bounds = [tuple(len(b) * [0.]), tuple(len(b) * [np.inf])])
    if not res.success:
        raise RuntimeError('Minimization failed: status = %s, message = \'%s\'' % (res.status, res.message))
    bhh = np.array(res.x)

    # Calculate our significance
    Z0 = np.sqrt(-2. * (n * np.log(np.sum(bhh) / n) + n - np.sum(bhh + 0.5 * ((b - bhh) / sigma) ** 2)))
    return Z0

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_data_from_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}
    return data

def dump_data_to_pickle(data, path):
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            data = pickle.dump(data, f)
        pass
    pass

def main():

    basedir   = os.path.dirname(os.path.abspath(__file__))
    pickledir = makedir(pjoin(basedir, 'pickles/'))
    plotdir   = makedir(pjoin(basedir, 'plots/'))

    #####################
    ### SECTION 2.1.1 ###
    #####################

    def Section2p1p1():

        s     = 50.
        b1    = 100.
        b2    = 50.
        tau11 = 60.
        tau22 = 40.
        tau12 = np.linspace(0., b1 * tau11 / b2, 100)
        tau21 = np.linspace(0., b2 * tau22 / b1, 100)

        z0 = np.empty((len(tau12), len(tau21)))
        for i in range(len(tau12)):
            for j in range(len(tau21)):
                z0[i, j] = nCRZ0(s, [b1, b2], [[tau11, tau12[i]], [tau21[j], tau22]])

        fig  = plt.figure()
        ax   = fig.add_subplot(111)
        pcm  = ax.pcolormesh(tau12 * b2 / (tau11 * b1), tau21 * b1 / (tau22 * b2), z0, cmap = 'magma', shading = 'nearest')
        pcm.set_edgecolor('face')
        cbar = plt.colorbar(pcm)
        ax.set_xlabel('($b_2$ in CR 1) / ($b_1$ in CR 1) [a.u.]')
        ax.set_ylabel('($b_1$ in CR 2) / ($b_2$ in CR 2) [a.u.]')
        cbar.set_label('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]', rotation = 270, labelpad = 20)
        # ax.set_title('Asymptotic significance for CRs with mixed background processes', pad = 10)
        plt.savefig(pjoin(plotdir, '1SRNCR_mixed_processes.eps'), format = 'eps', dpi = 1200)
        plt.close()

        multi = logVector(1, 1000, 100)

        z0 = np.empty((len(multi), len(multi)))
        for i in range(len(multi)):
            for j in range(len(multi)):
                z0[i, j] = nCRZ0(s, [b1, b2], [[multi[i], 0.1 * multi[i] * b1 / b2], [0.1 * multi[j] * b2 / b1, multi[j]]])

        fig  = plt.figure()
        ax   = fig.add_subplot(111)
        pcm  = ax.pcolormesh(multi, multi, z0, cmap = 'magma', shading = 'nearest')
        pcm.set_edgecolor('face')
        cbar = plt.colorbar(pcm)
        ax.set_xlabel('($b_1$ in SR) / ($b_1$ in CR 1) [a.u.]')
        ax.set_ylabel('($b_2$ in SR) / ($b_2$ in CR 2) [a.u.]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        cbar.set_label('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]', rotation = 270, labelpad = 20)
        # ax.set_title('Asymptotic significance for CRs varying transfer factors', pad = 10)
        plt.savefig(pjoin(plotdir, '1SRNCR_varying_tau.eps'), format = 'eps', dpi = 1200)
        plt.close()

    #####################
    ### SECTION 2.1.2 ###
    #####################

    def Section2p1p2():

        # Set the seed
        np.random.seed(43)

        datapath = pjoin(pickledir, 'Section2p1p2.pkl')
        s        = 10.
        b1       = [round(n) for n in logVector(1., 1000., 10)]
        b2       = [5., 25., 150.]
        tau1     = 8.
        tau2     = 5.
        colours  = ['g', 'b', 'r']

        data = load_data_from_pickle(datapath)

        for _b2, c in zip(b2, colours):
            k = str(int(_b2))
            if not data.get(k, {}):
                data[k] = {'z0': [], 't0': [], 't1': []}
                for _b1 in b1:
                    logger.info('On (b1, b2) = (%s, %s).' % (int(_b1), int(_b2)))
                    z0, t0, t1 = nCRZ0_MC(s, [_b1, _b2], [[tau1, 0.], [0., tau2]], return_t0_and_t1 = True, sleep = 0.001, ntoys = 50000)
                    data[k]['z0'].append(z0)
                    data[k]['t0'].append(t0)
                    data[k]['t1'].append(t1)
            plt.plot(b1, data[k]['z0'], marker = 'o', color = c, linewidth = 0, label = 'Numerical: $b_2 = %s$' % int(_b2))
            b1Fine = logVector(b1[0], b1[-1], 1000)
            plt.plot(b1Fine, [nCRZ0_DiagTau(s, [_b1, _b2], [tau1, tau2]) for _b1 in b1Fine], linestyle = '-', markersize = 0, color = c, label = 'Asymptotic: $b_2 = %s$' % int(_b2))
            plt.plot(b1Fine, s / np.sqrt(s + b1Fine + _b2), linestyle = '--', markersize = 0, color = c, label = 'Simple: $b_2 = %s$' % int(_b2))
        plt.xlim((b1[0], b1[-1]))
        plt.ylim((0., 3.5))
        plt.xlabel('Background 1 yield in SR $b_1$ [a.u.]')
        plt.ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
        plt.xscale('log')
        # plt.title('1 SR + 2 CRs Asymptotic Significance: $s = %s$, $\\tau_1 = %s$, $\\tau_2 = %s$' % (int(s), int(tau1), int(tau2)))
        plt.legend(loc = 'upper right')
        plt.savefig(pjoin(plotdir, '1SRplus2CR.eps'), format = 'eps', dpi = 1200)
        plt.close()

        axrange = (0., 25.)
        bins  = 100
        for _b1 in [1., 1000.]:
            t0, t1 = data['5']['t0'][b1.index(_b1)], data['5']['t1'][b1.index(_b1)]
            plt.hist(t0, weights = len(t0) * [1. / len(t0)], range = axrange, bins = bins, histtype = 'step', color = 'b', label = '$f(t_0|\\mu^\\prime = 0)$')
            plt.hist(t1, weights = len(t1) * [1. / len(t1)], range = axrange, bins = bins, histtype = 'step', color = 'r', label = '$f(t_0|\\mu^\\prime = 1)$')
            plt.xlim(axrange)
            plt.xlabel('Test statistic $t_0$ [a.u.]')
            plt.ylabel('Normalized counts [a.u.]')
            plt.yscale('log')
            plt.legend()
            plt.savefig(pjoin(plotdir, '1SRplus2CR_b1eq%s.eps' % int(_b1)), format = 'eps', dpi = 1200)
            plt.close()

        dump_data_to_pickle(data, datapath)

    #####################
    ### SECTION 2.2.1 ###
    #####################

    def Section2p2p1():

        # Set the seed
        np.random.seed(44)

        datapath = pjoin(pickledir, 'Section2p2p1.pkl')
        s1       = [round(n) for n in logVector(1., 100., 10)]
        s2       = [25., 10., 10.]
        s3       = 12.
        b        = [1000., 1000., 3000.]
        tau1     = 2.
        tau2     = 10.
        tau3     = 20.
        colours  = ['g', 'b', 'r']

        data = load_data_from_pickle(datapath)

        for _s2, _b, c in zip(s2, b, colours):
            k = str(int(_s2)) + '_' + str(int(_b))
            if not data.get(k, {}):
                data[k] = {'z0': [], 't0': [], 't1': []}
                for _s1 in s1:
                    logger.info('On (s1, s2, b) = (%s, %s, %s).' % (int(_s1), int(_s2), int(_b)))
                    ntoys = 100000 if (_s1 > 75.) else 50000
                    logger.info('Using %s toys.' % ntoys)
                    z0, t0, t1 = nSRZ0_MC([_s1, _s2, s3], _b, [tau1, tau2, tau3], return_t0_and_t1 = True, sleep = 0.001, ntoys = ntoys)
                    data[k]['z0'].append(z0)
                    data[k]['t0'].append(t0)
                    data[k]['t1'].append(t1)
            plt.plot(s1, data[k]['z0'], marker = 'o', color = c, linewidth = 0, label = 'Numerical: $(s_2, b) = (%s, %s)$' % (int(_s2), int(_b)))
            s1Fine = logVector(s1[0], s1[-1], 1000)
            plt.plot(s1Fine, [nSRZ0([_s1, _s2, s3], _b, [tau1, tau2, tau3]) for _s1 in s1Fine], linestyle = '-', markersize = 0, color = c, label = 'Asymptotic: $(s_2, b) = (%s, %s)$' % (int(_s2), int(_b)))
            plt.plot(s1Fine, np.sqrt((s1Fine / np.sqrt(s1Fine + _b / tau1)) ** 2 + (_s2 / np.sqrt(_s2 + _b / tau2)) ** 2 + (s3 / np.sqrt(s3 + _b / tau3)) ** 2), linestyle = '--', markersize = 0, color = c, label = 'Simple: $(s_2, b) = (%s, %s)$' % (int(_s2), int(_b)))
        plt.xlim((s1[0], s1[-1]))
        plt.ylim((0., 5.0))
        plt.xlabel('Signal yield in SR 1 $s_1$ [a.u.]')
        plt.ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
        plt.xscale('log')
        # plt.title('3 SRs + 1 CR Asymptotic Significance: $s_3 = %s$, $\\tau_1 = %s$, $\\tau_2 = %s$, $\\tau_3 = %s$' % (int(s3), int(tau1), int(tau2), int(tau3)))
        plt.legend(loc = 'upper left', bbox_to_anchor = (1.0, 1.02))
        plt.savefig(pjoin(plotdir, '3SRplus1CR.eps'), format = 'eps', dpi = 1200, bbox_inches = 'tight')
        plt.close()

        dump_data_to_pickle(data, datapath)

    #####################
    ### SECTION 2.4.2 ###
    #####################

    def Section2p4p2_vsB1():

        # Set the seed
        np.random.seed(45)

        datapath = pjoin(pickledir, 'Section2p4p2_vsB1.pkl')
        sigma1   = 5.
        sigma2   = 10.
        s        = 10.
        b1       = [round(n) for n in logVector(1., 1000., 10)]
        b2       = [5., 25., 150.]
        R        = [[lambda th: 1. + sigma1 / 100. * th, lambda th: 1.], [lambda th: 1., lambda th: 1. + sigma2 / 100. * th]]
        S        = [[1., 0.], [0., 1.]]
        colours  = ['g', 'b', 'r']

        data = load_data_from_pickle(datapath)

        for _b2, c in zip(b2, colours):
            k = str(int(_b2))
            if not data.get(k, {}):
                data[k] = {'z0': [], 't0': [], 't1': []}
                for _b1 in b1:
                    logger.info('On (b1, b2) = (%s, %s).' % (int(_b1), int(_b2)))
                    z0, t0, t1 = GaussZ0_MC(s, [_b1, _b2], R, S, return_t0_and_t1 = True, sleep = 0.001, ntoys = 50000)
                    data[k]['z0'].append(z0)
                    data[k]['t0'].append(t0)
                    data[k]['t1'].append(t1)
            plt.plot(b1, data[k]['z0'], marker = 'o', color = c, linewidth = 0, label = 'Numerical: $b_2 = %s$' % int(_b2))
            b1Fine = logVector(b1[0], b1[-1], 1000)
            plt.plot(b1Fine, [GaussZ0_Decorr(s, [_b1, _b2], [_b1 * sigma1 / 100., _b2 * sigma2 / 100.]) for _b1 in b1Fine], linestyle = '-', markersize = 0, color = c, label = 'Asymptotic: $b_2 = %s$' % int(_b2))
            plt.plot(b1Fine, s / np.sqrt(s + b1Fine + _b2 + (sigma1 / 100. * b1Fine) ** 2 + (sigma2 / 100. * _b2) ** 2), linestyle = '--', markersize = 0, color = c, label = 'Simple: $b_2 = %s$' % int(_b2))
        plt.xlim((b1[0], b1[-1]))
        plt.ylim((0., 3.5))
        plt.xlabel('Background 1 yield in SR $b_1$ [a.u.]')
        plt.ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
        plt.xscale('log')
        # plt.title('1 SR + 2 Gaussian Decorrelated Constraints Asymptotic Significance:\n$s = {}$, $\\sigma_1 = {}\\%$, $\\sigma_2 = {}\\%$'.format(int(s), int(sigma1), int(sigma2)))
        plt.legend(loc = 'upper right')
        plt.savefig(pjoin(plotdir, '1SRplus2GaussConst.eps'), format = 'eps', dpi = 1200)
        plt.close()

        dump_data_to_pickle(data, datapath)

    def Section2p4p2_vsSigma():

        sigma1   = np.hstack([logVector(0.1, 100., 15), logVector(100., 400., 3)[1:]])
        sigma2   = [1., 10., 100.]
        s        = 10.
        b1       = [25., 50., 50., 150.]
        b2       = [25., 50., 150., 50.]
        colours  = ['gold', 'g', 'b', 'r']

        fig, axs = plt.subplots(nrows = 2, ncols = 2, sharex = 'col', sharey = 'row', figsize = [2 * 6.0, 2 * 4.0])
        axs[1, 1].axis('off')

        for i, _sigma2 in enumerate(sigma2):

            # Set the seed - let's use a fresh seed on each loop iteration, as we are saving separate pickles
            # (this allows us to cleanly reproduce the results, per pickle, without throwing all of the toys in previous)
            np.random.seed(60 + i)

            # Dump a pickle for each sigma2 loop
            datapath = pjoin(pickledir, 'Section2p4p2_vsSigma_sigma2eq%s.pkl' % int(_sigma2))
            data     = load_data_from_pickle(datapath)

            if i == 0:
                ax = axs[0, 0]
            elif i == 1:
                ax = axs[0, 1]
            elif i == 2:
                ax = axs[1, 0]
            elif i == 3:
                continue
            else:
                ax = None

            for _b1, _b2, c in zip(b1, b2, colours):
                k = str(int(_b1)) + '_' + str(int(_b2))
                if not data.get(k, {}):
                    data[k] = {'z0': [], 't0': [], 't1': []}
                    for _sigma1 in sigma1:
                        logger.info('On (sigma1, sigma2, b1, b2) = (%s, %s, %s, %s).' % (round(_sigma1, 5), round(_sigma2, 5), int(_b1), int(_b2)))
                        z0, t0, t1 = GaussZ0_MC(s, [_b1, _b2], R(_sigma1, _sigma2), S, return_t0_and_t1 = True, sleep = 0.001, ntoys = 50000, retry_first = False, skip_failed_toys = True)
                        data[k]['z0'].append(z0)
                        data[k]['t0'].append(t0)
                        data[k]['t1'].append(t1)
                ax.plot(sigma1, data[k]['z0'], marker = 'o', color = c, linewidth = 0, label = 'Numerical: $(b_1, b_2) = (%s, %s)$' % (int(_b1), int(_b2)) if i == 0 else '')
                sigma1Fine = logVector(sigma1[0], sigma1[-1] if sigma1[-1] > 1000. else 1000., 1000)
                ax.plot(sigma1Fine, [GaussZ0_Decorr(s, [_b1, _b2], [_b1 * _sigma1 / 100., _b2 * _sigma2 / 100.]) for _sigma1 in sigma1Fine], linestyle = '-', markersize = 0, color = c, label = 'Asymptotic: $(b_1, b_2) = (%s, %s)$' % (int(_b1), int(_b2)) if i == 0 else '')
                ax.plot(sigma1Fine, s / np.sqrt(s + _b1 + _b2 + (sigma1Fine / 100. * _b1) ** 2 + (_sigma2 / 100. * _b2) ** 2), linestyle = '--', markersize = 0, color = c, label = 'Simple: $(b_1, b_2) = (%s, %s)$' % (int(_b1), int(_b2)) if i == 0 else '')
            ax.set_ylim((0., 1.4))
            if i != 1: ax.set_ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
            ax.text(40, 1.2, '$s = {}$, $\\sigma_2 = {}\\%$'.format(int(s), int(_sigma2)), fontsize = 12, bbox = {'facecolor': 'white', 'pad': 10})
            if i != 0:
                ax.set_xlim((sigma1[0], sigma1[-1] if sigma1[-1] > 1000. else 1000.))
                ax.set_xlabel('Background 1 yield uncertainty in SR $\\sigma_1$ [\\%]')
                ax.set_xscale('log')
            if i == 1: ax.xaxis.set_tick_params(labelbottom = True)
            dump_data_to_pickle(data, datapath)

        # fig.suptitle('1 SR + 2 Decorrelated Gaussian Constraints Asymptotic Significance')
        axs[0, 0].legend(loc = 'upper left', bbox_to_anchor = (1.05, -0.15))
        plt.subplots_adjust(hspace = 0.05, wspace = 0.05) # , top = 0.95, bottom = 0.05)
        plt.savefig(pjoin(plotdir, '1SRplus2GaussConst_err.eps'), format = 'eps', dpi = 1200, bbox_inches = 'tight')
        plt.close()

    #####################
    ### SECTION 2.4.4 ###
    #####################

    def Section2p4p4_Corr():

        # Set the seed
        np.random.seed(47)

        datapath = pjoin(pickledir, 'Section2p4p4_Corr.pkl')
        s        = 10.
        b1       = [round(n) for n in logVector(1., 1000., 10)]
        b2       = 5.
        sigma1   = 35.
        sigma2   = 70.
        R        = [[lambda th: 1. + sigma1 / 100. * th, lambda th: 1.], [lambda th: 1., lambda th: 1. + sigma2 / 100. * th]]
        S        = [[1., 0.75], [0.75, 1.]]

        data = load_data_from_pickle(datapath)

        if not all(data.get(k, []) for k in ['z0', 't0', 't1']):
            data.update({'z0': [], 't0': [], 't1': []})
            for _b1 in b1:
                logger.info('On b1 = %s.' % int(_b1))
                z0, t0, t1 = GaussZ0_MC(s, [_b1, b2], R, S, return_t0_and_t1 = True, sleep = 0.002, ntoys = 50000)
                data['z0'].append(z0)
                data['t0'].append(t0)
                data['t1'].append(t1)
        plt.plot(b1, data['z0'], marker = 'o', color = 'r', linewidth = 0, label = 'Numerical')
        b1Fine = logVector(b1[0], b1[-1], 1000)
        plt.plot(b1Fine, [GaussZ0(s = s, b = [_b1, b2], R = R, S = S) for _b1 in b1Fine], linestyle = '-', markersize = 0, color = 'r', label = 'Asymptotic (corr.)')
        plt.plot(b1Fine, [GaussZ0(s = s, b = [_b1, b2], R = R, S = [[1., 0.], [0., 1.]]) for _b1 in b1Fine], linestyle = ':', markersize = 0, color = 'darkred', label = 'Asymptotic (decorr.)')
        plt.plot(b1Fine, s / np.sqrt(s + b1Fine + b2 + (sigma1 / 100. * b1Fine) ** 2 + (sigma2 / 100. * b2) ** 2), linestyle = '--', markersize = 0, color = 'lightcoral', label = 'Simple')
        plt.xlim((b1[0], b1[-1]))
        plt.ylim((0., 2.))
        plt.xlabel('Background 1 yield in SR $b_1$ [a.u.]')
        plt.ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
        plt.xscale('log')
        # plt.title('1 SR + 2 Gaussian Correlated Constraints Asymptotic Significance:\n$s = {}$, $b_2 = {}$, $\\sigma_1 = {}\\%$, $\\sigma_2 = {}\\%$'.format(int(s), int(b2), int(sigma1), int(sigma2)))
        plt.legend(loc = 'upper right')
        plt.savefig(pjoin(plotdir, '1SRplus2GaussConst_corr.eps'), format = 'eps', dpi = 1200)
        plt.close()

        dump_data_to_pickle(data, datapath)

    #####################
    ### SECTION 2.4.5 ###
    #####################

    def Section2p4p5_Response():

        # Set the seed
        np.random.seed(49)

        def smooth_interpolate(th, func1, func2, weight):
            return weight(th) * func1(th) + (1. - weight(th)) * func2(th)

        def heaviside(th, sigma_lo, sigma_hi):
            return smooth_interpolate(th, lambda th: 1. + sigma_lo * th, lambda th: 1. + sigma_hi * th, lambda th: 1. - np.heaviside(th, 1.))

        def arctan(th, sigma_lo, sigma_hi, k = 10.):
            return smooth_interpolate(th, lambda th: 1. + sigma_lo * th, lambda th: 1. + sigma_hi * th, lambda th: (1. - 2. / np.pi * np.arctan(np.pi / 2. * k * th)) / 2.)

        def tanh(th, sigma_lo, sigma_hi, k = 10.):
            return smooth_interpolate(th, lambda th: 1. + sigma_lo * th, lambda th: 1. + sigma_hi * th, lambda th: (1. - np.tanh(k * th)) / 2.)

        def erf(th, sigma_lo, sigma_hi, k = 10.):
            return smooth_interpolate(th, lambda th: 1. + sigma_lo * th, lambda th: 1. + sigma_hi * th, lambda th: (1. - scipy.special.erf(k * th)) / 2.)

        def sigmoid(th, sigma_lo, sigma_hi, k = 10.):
            return smooth_interpolate(th, lambda th: 1. + sigma_lo * th, lambda th: 1. + sigma_hi * th, lambda th: 1. - 1. / (1. + np.exp(-k * th)))

        response_functions = {'Heaviside': (heaviside, 'k', '-'), 'arctan': (arctan, 'g', '--'), 'tanh': (tanh, 'b', ':'), 'erf': (erf, 'r', '-.'), 'sigmoid': (sigmoid, 'gold', '-')}

        sigma_lo = 0.20
        sigma_hi = 0.35
        th = np.linspace(-1., +1., 1000)
        for l, (f, c, ls) in response_functions.items():
            plt.plot(th, f(th, sigma_lo, sigma_hi), color = c, label = l, linestyle = ls)
        plt.xlim((th[0], th[-1]))
        plt.ylim((1. - sigma_lo, 1. + sigma_hi))
        plt.xlabel('Nuisance parameter $\\theta$ [a.u.]')
        plt.ylabel('Response function $R(\\theta)$ [a.u.]')
        # plt.title('Different Response Functions')
        plt.legend(loc = 'upper left')
        plt.savefig(pjoin(plotdir, 'response_functions.eps'), format = 'eps', dpi = 1200)
        plt.xlim((-0.2, +0.2))
        plt.ylim((0.95, 1.075))
        plt.savefig(pjoin(plotdir, 'response_functions_zoomed.eps'), format = 'eps', dpi = 1200)
        plt.close()

        # 1st derivatives:
        th = np.linspace(-1., +1., 1000)
        for l, (f, c, ls) in response_functions.items():
            plt.plot(th, scipy.misc.derivative(lambda th: f(th, sigma_lo, sigma_hi), th, dx = 1e-6), color = c, label = l, linestyle = ls)
        plt.xlim((th[0], th[-1]))
        plt.ylim((0.15, 0.40))
        plt.xlabel('Nuisance parameter $\\theta$ [a.u.]')
        plt.ylabel('Derivative of response function $dR(\\theta)/d\\theta$ [a.u.]')
        # plt.title('Dervatives of Different Response Functions')
        plt.legend(loc = 'upper left')
        plt.savefig(pjoin(plotdir, 'response_functions_derivatives.eps'), format = 'eps', dpi = 1200)
        plt.close()

        s         = 10.
        b1        = logVector(1., 10000., 100)
        b2        = 5.
        sigma1_lo = 20. / 100.
        sigma1_hi = 35. / 100.
        sigma2_lo = 70. / 100.
        sigma2_hi = 90. / 100.
        R         = lambda sigma1_lo, sigma1_hi, sigma2_lo, sigma2_hi: [[lambda th: f(th, sigma1_lo, sigma1_hi), lambda th: 1.], [lambda th: 1., lambda th: f(th, sigma2_lo, sigma2_hi)]]
        S         = [[1., 0.75], [0.75, 1.]]

        for l, (f, c, ls) in response_functions.items():
            plt.plot(b1, [GaussZ0(s = s, b = [_b1, b2], R = R(sigma1_lo, sigma1_hi, sigma2_lo, sigma2_hi), S = S) for _b1 in b1], linestyle = ls, markersize = 0, color = c, label = l)

        plt.xlim((b1[0], b1[-1]))
        plt.ylim((0.001, 10.))
        plt.xlabel('Background 1 yield in SR $b_1$ [a.u.]')
        plt.ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
        plt.xscale('log')
        plt.yscale('log')
        # plt.title('Sensitivities for Different Response Functions:\n$s = {}$, $b_2 = {}$'.format(int(s), int(b2)))
        plt.legend(loc = 'upper right')
        plt.savefig(pjoin(plotdir, 'response_functions_z0_b2eq%s.eps' % int(b2)), format = 'eps', dpi = 1200, bbox_inches = 'tight')
        plt.close()

        s  = 100.
        b2 = 10000.

        for l, (f, c, ls) in response_functions.items():
            plt.plot(b1, [GaussZ0(s = s, b = [_b1, b2], R = R(sigma1_lo, sigma1_hi, sigma2_lo, sigma2_hi), S = S) for _b1 in b1], linestyle = ls, markersize = 0, color = c, label = l)

        plt.xlim((b1[0], b1[-1]))
        plt.ylim((0.008, 0.02))
        plt.xlabel('Background 1 yield in SR $b_1$ [a.u.]')
        plt.ylabel('Significance of discovery $\\textrm{med}[Z_0|\\mu^\\prime=1]$ [a.u.]')
        plt.xscale('log')
        plt.yscale('log')
        # plt.title('Sensitivities for Different Response Functions:\n$s = {}$, $b_2 = {}$'.format(int(s), int(b2)))
        plt.legend(loc = 'upper right')
        plt.savefig(pjoin(plotdir, 'response_functions_z0_b2eq%s.eps' % int(b2)), format = 'eps', dpi = 1200, bbox_inches = 'tight')
        plt.close()

    #####################
    ### SECTION 2.4.6 ###
    #####################

    def Section2p4p6_CPU():

        # Set the seed
        np.random.seed(48)

        datapath = pjoin(pickledir, 'Section2p4p6_CPU.pkl')
        s        = 10.
        b1       = 10.
        b2       = 5.
        sigma1   = 35.
        sigma2   = 70.
        R        = [[lambda th: 1. + sigma1 / 100. * th, lambda th: 1.], [lambda th: 1., lambda th: 1. + sigma2 / 100. * th]]
        S        = [[1., 0.75], [0.75, 1.]]
        ntoys    = [round(n) for n in logVector(1000, 1000000, 40)]

        data = load_data_from_pickle(datapath)

        if not all(data.get(k, []) for k in ['z0', 't0', 't1', 'cpu']):
            data.update({'z0': [], 't0': [], 't1': [], 'cpu': []})
            for _ntoys in ntoys:
                logger.info('On ntoys = %s.' % int(_ntoys))
                logging.getLogger(asymptotic_formulae.__name__).setLevel(level = logging.WARNING)
                start = time.clock()
                z0, t0, t1 = GaussZ0_MC(s, [b1, b2], R, S, return_t0_and_t1 = True, sleep = 0.001, ntoys = _ntoys, retry_first = False, skip_failed_toys = True)
                stop = time.clock()
                logging.getLogger(asymptotic_formulae.__name__).setLevel(level = logging.DEBUG)
                data['z0'].append(z0)
                data['t0'].append(t0)
                data['t1'].append(t1)
                delta = stop - start
                logger.info('Z0 = %s, CPU time = %s s.' % (z0, delta))
                data['cpu'].append(delta)

        if not all(data.get(k, []) for k in ['cpu_asymptotic', 'z0_asymptotic']):
            data['cpu_asymptotic'] = []
            data['z0_asymptotic'] = []
            for i in range(len(ntoys)):
                logger.info('On iteration %s.' % i)
                logging.getLogger(asymptotic_formulae.__name__).setLevel(level = logging.WARNING)
                start = time.clock()
                z0 = GaussZ0(s = s, b = [b1, b2], R = R, S = S)
                stop = time.clock()
                logging.getLogger(asymptotic_formulae.__name__).setLevel(level = logging.DEBUG)
                delta = stop - start
                logger.info('CPU time = %s s.' % delta)
                data['cpu_asymptotic'].append(delta)
                data['z0_asymptotic'].append(z0)
        z0 = GaussZ0(s = s, b = [b1, b2], R = R, S = S)

        fig = plt.figure()
        fig, axs = plt.subplots(2, 1, sharex = True)
        fig.subplots_adjust(hspace = 0.1)
        # fig.suptitle('CPU Comparisons: Numerical vs. Asymptotic for Gaussian Constraints')
        axs[0].plot(ntoys, data['z0'], color = 'darkorange', label = 'Numerical')
        axs[0].plot(ntoys, data['z0_asymptotic'], color = 'navy', label = 'Asymptotic')
        axs[0].set_ylabel('Significance of discovery [a.u.]')
        axs[0].set_ylim((1.15, 1.30))
        axs[0].legend(loc = 'upper right')
        axs[1].plot(ntoys, data['cpu'], color = 'darkorange', label = 'Numerical')
        axs[1].plot(ntoys, data['cpu_asymptotic'], color = 'navy', label = 'Asymptotic')
        axs[1].set_xlabel('Number of toys [a.u.]')
        axs[1].set_ylabel('CPU time [s]')
        axs[1].set_xlim((ntoys[0], ntoys[-1]))
        axs[1].set_ylim((1e-3, 1e4))
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        plt.savefig(pjoin(plotdir, 'Section2p4p2_CPU.eps'), format = 'eps', dpi = 1200)
        plt.close()

        dump_data_to_pickle(data, datapath)

    Section2p1p1()
    Section2p1p2()
    Section2p2p1()
    Section2p4p2_vsB1()
    Section2p4p2_vsSigma()
    Section2p4p4_Corr()
    Section2p4p5_Response()
    Section2p4p6_CPU()

if __name__ == '__main__':
    main()
