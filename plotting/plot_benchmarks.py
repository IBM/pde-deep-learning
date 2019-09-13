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


def plot_cost():
    files = [
        '190911-125255_Demo.txt',
        '190911-192253_Demo.txt'
    ]

    img_path = '../output/img/'
    file_path = '../output/benchmarks/'
    label = '$\lambda$, $\kappa$: '

    cost = {}
    file_num = 0
    for file in files:
        with open(file_path + file) as f:
            header = f.readline()
            lines = f.readlines()

        head = header.rstrip().split('\t')
        gamma_ix = head.index('cc_coeff')
        kappa_ix = head.index('kappa')
        iter_ix = head.index('iter')
        mse_ix = head.index('MSE')
        mae_ix = head.index('MAE')
        smape_ix = head.index('sMAPE')

        for line in lines:
            row = line.rstrip().split('\t')
            g = float(row[gamma_ix])
            ek = float(row[kappa_ix])
            iteration = int(row[iter_ix])
            # if iteration == 1:
            #     file_num += 1
            if (g, ek, file_num) not in cost:
                cost[(g, ek, file_num)] = defaultdict(list)
            mse = float(row[mse_ix])
            mae = float(row[mae_ix])
            smape = float(row[smape_ix])
            cost[(g, ek, file_num)][iteration].append((mse, mae, smape))

    cost_plot = {key: [[i] + list(np.average(np.transpose(tup), axis=1))
                       + list(np.std(np.transpose(tup), axis=1))
                       for i, tup in iters.items()]
                 for key, iters in cost.items()}

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_yscale("log", nonposy='clip')
    handle = {}
    for (g, ek, n), triples in cost_plot.items():
        x, yl, ye = np.transpose([[d[0], d[1], d[3]] for d in triples])
        handle[(g, ek, n)] = plt.errorbar(x, yl, yerr=ye, label=label + str(g) + ', ' + str(ek))
    plt.xlabel('iterations')
    plt.ylabel('mean absolute error')
    plt.ylim(0.0001, 1)
    plt.grid(True)
    plt.legend(handles=list(handle.values()))
    ax.set_title('mean absolute error')
    # fig.subplots_adjust(top=0.85, hspace=0.3)
    plt.savefig(img_path + 'benchmarks/mae_' + str(files) + '.pdf')
    plt.savefig(img_path + 'benchmarks/mae_' + str(files) + '.png')
    plt.close()


if __name__ == '__main__':
    plot_cost()
