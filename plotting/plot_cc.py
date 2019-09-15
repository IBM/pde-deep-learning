from collections import defaultdict

#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt

import numpy as np


""" 

-*- coding: utf-8 -*-

Legal:
    (C) Copyright IBM 2018.

    This code is licensed under the Apache License, Version 2.0. You may
    obtain a copy of this license in the LICENSE.txt file in the root 
    directory of this source tree or at 
    http://www.apache.org/licenses/LICENSE-2.0.

    Any modifications or derivative works of this code must retain this
    copyright notice, and modified files need to carry a notice 
    indicating that they have been altered from the originals.

    IBM-Review-Requirement: Art30.3
    Please note that the following code was developed for the project 
    VaVeL at IBM Research -- Ireland, funded by the European Union under 
    the Horizon 2020 Program.
    The project started on December 1st, 2015 and was completed by 
    December 1st, 2018. Thus, in accordance with Article 30.3 of the 
    Multi-Beneficiary General Model Grant Agreement of the Program, 
    there are certain limitations in force  up to December 1st, 2022. 
    For further details please contact Jakub Marecek 
    (jakub.marecek@ie.ibm.com) or Gal Weiss (wgal@ie.ibm.com).

If you use the code, please cite our paper:
https://arxiv.org/abs/1810.09425

Authors: 
    Philipp Hähnel <phahnel@hsph.harvard.edu>

Last updated:
    2019 - 09 - 13

"""


def plot_CC():
    files = [
        '190911-125255_Demo.txt',
        '190911-192253_Demo.txt'
    ]

    img_path = '../output/img/'
    file_path = '../output/cc/'
    label = '$\lambda$, $\kappa$: '

    chi = {}
    for file in files:
        with open(file_path + file) as f:
            header = f.readline()
            lines = f.readlines()

        head = header.rstrip().split('\t')
        seed_ix = header.index('seed')
        gamma_ix = head.index('cc_coeff')
        kappa_ix = head.index('kappa')
        iter_ix = head.index('iter')
        chi_low_mean_ix = head.index('ch_l_avg')
        chi_high_mean_ix = head.index('ch_u_avg')
        chi_diff_ix = head.index('chi_avg_dist')
        Y_diff_ix = head.index('Y_avg_dist')
        version_ix = head.index('version')

        for line in lines:
            row = line.rstrip().split('\t')
            version = row[version_ix]
            seed = int(row[seed_ix])
            g = float(row[gamma_ix])
            ek = float(row[kappa_ix])
            if (g, ek, seed) not in chi:
                chi[(g, ek, seed)] = defaultdict(list)
            iteration = int(row[iter_ix])
            chi_diff = float(row[chi_diff_ix])
            Y_diff = float(row[Y_diff_ix])
            chi_low_mean = float(row[chi_low_mean_ix])
            chi_high_mean = float(row[chi_high_mean_ix])
            chi[(g, ek, seed)][iteration].append((chi_diff, Y_diff, chi_low_mean, chi_high_mean))

    chi_plot = {key: [[i] + list(np.average(np.transpose(tup), axis=1))
                          + list(np.std(np.transpose(tup), axis=1))
                      for i, tup in iters.items()]
                for key, iters in chi.items()}

    fig = plt.figure(figsize=(5, 5))
    handle = {}
    for (g, ek, n), triples in chi_plot.items():
        x, yl, ye = np.transpose([[d[0], d[1], d[5]] for d in triples])
        handle[(g, ek, n)] = plt.errorbar(x, yl, yerr=ye, label=label + str(g) + ', ' + str(ek))
    plt.xlabel('iterations')
    plt.ylabel('$|\chi - \chi|$')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(handles=list(handle.values()))
    plt.title('mean difference between upper and lower $\chi$ values')
    plt.savefig(img_path + 'benchmarks/chi_diff_' + version + '.pdf')
    plt.savefig(img_path + 'benchmarks/chi_diff_' + version + '.png')
    plt.close()

    fig = plt.figure(figsize=(5, 5))
    handle = {}
    for (g, ek, n), triples in chi_plot.items():
        x, yl, ye = np.transpose([[d[0], d[3], d[7]] for d in triples])
        p = plt.errorbar(x, yl, yerr=ye)
        x, yl, ye = np.transpose([[d[0], d[4], d[8]] for d in triples])
        handle[(g, ek, n)] = plt.errorbar(x, yl, yerr=ye, label=label + str(g) + ', ' + str(ek), color=p[0].get_color())
    plt.xlabel('iterations')
    plt.ylabel('mean $\chi$')
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.legend(handles=list(handle.values()))
    plt.title('mean upper and lower $\chi$ values')
    plt.savefig(img_path + 'benchmarks/chi_mean_interval_' + version + '.pdf')
    plt.savefig(img_path + 'benchmarks/chi_mean_interval_' + version + '.png')
    plt.close()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_yscale("log", nonposy='clip')
    handle = {}
    for (g, ek, n), triples in chi_plot.items():
        x, yh, ye = np.transpose([[d[0], d[2], d[6]] for d in triples])
        handle[(g, ek, n)] = plt.errorbar(x, yh, yerr=ye, label=label + str(g) + ', ' + str(ek))
    ax.set_xlabel('iterations')
    ax.set_ylabel('|f$_{upper}$ - f$_{lower}$|')
    plt.ylim(0.01, 2)
    plt.grid(True)
    plt.legend(handles=list(handle.values()))
    ax.set_title('mean difference between upper and lower predictions')
    plt.savefig(img_path + 'benchmarks/Y_diff_' + version + '.pdf')
    plt.savefig(img_path + 'benchmarks/Y_diff_' + version + '.png')
    plt.close()

    return None


if __name__ == '__main__':
    plot_CC()
