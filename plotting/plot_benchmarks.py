from collections import defaultdict

from matplotlib import rc
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Helvetica'
#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    2020 - 01 - 15

"""


def plot_cost_and_cc():
    case = 'Demo'
    if case == 'Demo':
        files = [
            # '190911-125255_Demo.txt',
            # '190911-192253_Demo.txt',
            # '190915-171120_Demo.txt',
            # '190915-222342_Demo.txt',
            # '191218-082528_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.88
            # '191218-172926_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.88
            # '191219-051306_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.88
            # '191219-141352_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.88
            # '191219-233123_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.88
            # '191220-150848_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.88
            # '191221-103107_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.88
            # '191221-201931_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.88
            # '191222-085443_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.88
            # '191222-175315_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.88
            # '191223-114646_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.88
            # '191223-203546_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.88
            # '191224-113152_Demo.txt',  # l=0, k=1, lr=3e-4, df=0.86
            # '191224-225739_Demo.txt',  # l=0, k=0.1, lr=3e-4, df=0.86
            '200103-105225_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.85
            '200103-192358_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.85
            '200104-040243_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.85
            '200104-125543_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.85
            '200104-221710_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.85
            '200105-082422_Demo.txt',  # l=0, k=0, lr=1e-4, df=0.85
            '191225-113612_Demo.txt',  # l=0, k=0.1, lr=1e-4, df=0.85
            '191225-210136_Demo.txt',  # l=0, k=0.1, lr=1e-4, df=0.85
            '191226-053515_Demo.txt',  # l=0, k=0.1, lr=1e-4, df=0.85
            '191226-141149_Demo.txt',  # l=0, k=0.1, lr=1e-4, df=0.85
            '191226-230604_Demo.txt',  # l=0, k=0.1, lr=1e-4, df=0.85
            '191227-081640_Demo.txt',  # l=0, k=0.1, lr=1e-4, df=0.85
            '200101-111625_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.85
            '200101-194645_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.85
            '200102-044653_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.85
            '200102-135135_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.85
            '200102-234547_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.85
            '200106-175520_Demo.txt',  # l=0, k=0.5, lr=1e-4, df=0.85
            '191227-194307_Demo.txt',  # l=1, k=0.1, lr=1e-4, df=0.85
            '191228-041239_Demo.txt',  # l=1, k=0.1, lr=1e-4, df=0.85
            '191228-125703_Demo.txt',  # l=1, k=0.1, lr=1e-4, df=0.85
            '191228-221233_Demo.txt',  # l=1, k=0.1, lr=1e-4, df=0.85
            '200105-231403_Demo.txt',  # l=1, k=0.1, lr=1e-4, df=0.85
            '200106-082125_Demo.txt',  # l=1, k=0.1, lr=1e-4, df=0.85
            '191230-002838_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.85
            '191230-085244_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.85
            '191230-180101_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.85
            '191231-035053_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.85
            '191231-133229_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.85
            '191231-235235_Demo.txt',  # l=1, k=0.5, lr=1e-4, df=0.85
            '200107-224655_Demo.txt',  # l=10, k=0.5, lr=1e-4, df=0.85
            '200108-221034_Demo.txt',  # l=10, k=0.5, lr=1e-4, df=0.85
            '200109-235238_Demo.txt',  # l=10, k=0.5, lr=1e-4, df=0.85
            '200110-183246_Demo.txt',  # l=10, k=0.5, lr=1e-4, df=0.85
            '200111-030819_Demo.txt',  # l=10, k=0.5, lr=1e-4, df=0.85
            '200111-115702_Demo.txt',  # l=10, k=0.5, lr=1e-4, df=0.85
        ]
    elif case == 'Dublin':
        files = [
            '190913-154541_Dublin.txt'
        ]
    else:
        files = []

    group_offset = 0.025

    ylim_mse = (0.001, 1)
    ylim_mae = (0.01, 1)
    ylim_smape = (0.4, 1)
    ylim_cc_diff = (0, 1.1)
    ylim_cc = (-0.3, 1.1)
    # ylim_y_diff = (0.01, 1)  # log plot
    ylim_y_diff = (0, 0.4)  # plot

    min_iter = 9  # ratios between parameter runs are considered for iterations >= min_iter

    individual_lw = 0.2
    individual_alpha = 0.6

    ###
    # plot cost data
    ###

    img_path = f'../output/img/benchmarks/{case}/'
    file_path = '../output/benchmarks/'
    label = r'$\lambda$, $\kappa$: '

    cost = {}
    cost_smape = {}
    file_num = 0
    for file in files:
        with open(file_path + file) as f:
            header = f.readline()
            lines = f.readlines()

        head = header.rstrip().split('\t')
        gamma_ix = head.index('cc_coeff')
        kappa_ix = head.index('kappa')
        tile_ix = head.index('tile')
        iter_ix = head.index('iter')
        mse_ix = head.index('MSE')
        mae_ix = head.index('MAE')
        smape_ix = head.index('sMAPE')

        for line in lines:
            row = line.rstrip().split('\t')
            g = float(row[gamma_ix])
            ek = float(row[kappa_ix])
            tile = int(row[tile_ix])
            iteration = int(row[iter_ix])
            # if iteration == 1:
            #     file_num += 1
            if (g, ek) not in cost:
                cost[(g, ek)] = defaultdict(list)
            if (g, ek, tile) not in cost_smape:
                cost_smape[(g, ek, tile)] = defaultdict(list)
            mse = float(row[mse_ix])
            mae = float(row[mae_ix])
            smape = float(row[smape_ix])
            cost[(g, ek)][iteration].append((mse, mae, smape))
            cost_smape[(g, ek, tile)][iteration].append((mse, mae, smape))

    # (iteration,
    # mse_mean, mae_mean, smape_mean,
    # mse_err, mae_err, smape_err)
    cost_plot = {key: [[i] + list(np.average(np.transpose(tup), axis=1))
                       + list(np.std(np.transpose(tup), axis=1))
                       for i, tup in iters.items()]
                 for key, iters in cost.items()}
    cost_plot = sorted(cost_plot.items(),
                       key=lambda kv_pair: (kv_pair[0][0], kv_pair[0][1]))
    cost_plot_smape = {key: [[i] + list(np.average(np.transpose(tup), axis=1))
                             + list(np.std(np.transpose(tup), axis=1))
                             for i, tup in iters.items()]
                       for key, iters in cost_smape.items()}
    cost_plot_smape = sorted(cost_plot_smape.items(),
                             key=lambda kv_pair: (kv_pair[0][2], kv_pair[0][0], kv_pair[0][1]))

    ###
    # plot MSE
    ###

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_yscale("log", nonposy='clip')
    handle = {}
    x = {}
    for i, ((g, ek), triples) in enumerate(cost_plot):
        x[i], yl, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[1], d[4]] for d in triples]
        )
        handle[(g, ek)] = plt.errorbar(
            x[i], yl, yerr=ye, label=label + str(g) + ', ' + str(ek)
        )
        # plot individual runs:
        cost_plot_individual = {
            i: np.transpose(tup)[0]
            for i, tup in cost[(g, ek)].items()
        }
        cost_individual = np.transpose([
            list(c) for c in cost_plot_individual.values()
        ])
        for ci in cost_individual:
            plt.plot(list(cost_plot_individual.keys()), ci,
                     color=handle[(g, ek)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
    ax.set_xticks(x[0])
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.ylim(*ylim_mse)
    plt.grid(lw=0.2, axis='y')
    plt.legend(handles=list(handle.values()))
    # ax.set_title('mean squared error')
    # fig.subplots_adjust(top=0.85, hspace=0.3)
    fig.subplots_adjust(left=0.14)
    plt.savefig(img_path + f'mse_{case}.pdf')
    # plt.savefig(img_path + f'mse_{case}.png')
    plt.close()

    ###
    # plot MAE
    ###

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.set_yscale("log", nonposy='clip')
    handle = {}
    x = {}
    for i, ((g, ek), triples) in enumerate(cost_plot):
        x[i], yl, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[2], d[5]] for d in triples]
        )
        handle[(g, ek)] = plt.errorbar(
            x[i], yl, yerr=ye, label=label + str(g) + ', ' + str(ek)
        )
        # plot individual runs:
        cost_plot_individual = {
            i: np.transpose(tup)[1]
            for i, tup in cost[(g, ek)].items()
        }
        cost_individual = np.transpose([
            list(c) for c in cost_plot_individual.values()
        ])
        for ci in cost_individual:
            plt.plot(list(cost_plot_individual.keys()), ci,
                     color=handle[(g, ek)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
    ax.set_xticks(x[0])
    plt.xlabel('Iteration')
    plt.ylabel('Mean Absolute Error')
    plt.ylim(*ylim_mae)
    plt.grid(lw=0.2, axis='y')
    plt.legend(handles=list(handle.values()))
    # ax.set_title('mean absolute error')
    # fig.subplots_adjust(top=0.85, hspace=0.3)
    fig.subplots_adjust(left=0.14)
    plt.savefig(img_path + f'mae_{case}.pdf')
    # plt.savefig(img_path + f'mae_{case}.png')
    plt.close()

    ###
    # plot sMAPE
    ###

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # ax.set_yscale("log", nonposy='clip')
    handle = {}
    x = {}
    for i, ((g, ek, t), triples) in enumerate(cost_plot_smape):
        x[i], yl, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[3], d[6]] for d in triples]
        )
        handle[(g, ek, t)] = plt.errorbar(
            x[i], yl, yerr=ye, label=f'{label} {g}, {ek}; tile: {t}'
        )
        # plot individual runs:
        cost_plot_individual = {
            i: np.transpose(tup)[2]
            for i, tup in cost_smape[(g, ek, t)].items()
        }
        cost_individual = np.transpose([
            list(c) for c in cost_plot_individual.values()
        ])
        for ci in cost_individual:
            plt.plot(list(cost_plot_individual.keys()), ci,
                     color=handle[(g, ek, t)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
    ax.set_xticks(x[0])
    plt.xlabel('Iteration')
    plt.ylabel('Symmetric Mean Absolute Percentage Error')
    plt.ylim(*ylim_smape)
    plt.grid(lw=0.2, axis='y')
    plt.legend(handles=list(handle.values()))
    # ax.set_title('Symmetric Mean Absolute Percentage Error')
    # fig.subplots_adjust(top=0.85, hspace=0.3)
    # fig.subplots_adjust(left=0.2)
    plt.savefig(img_path + f'smape_{case}.pdf')
    # plt.savefig(img_path + f'smape_{case}.png')
    plt.close()

    ###
    # plot consistency constraints data: chi, chi_diff, Y_diff
    ###

    file_path = '../output/cc/'
    version = ''

    chi = {}
    for file in files:
        with open(file_path + file) as f:
            header = f.readline()
            lines = f.readlines()

        head = header.rstrip().split('\t')
        seed_ix = head.index('seed')
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
            if (g, ek) not in chi:
                chi[(g, ek)] = defaultdict(list)
            iteration = int(row[iter_ix]) - 1
            chi_diff = float(row[chi_diff_ix])
            Y_diff = float(row[Y_diff_ix])
            chi_low_mean = float(row[chi_low_mean_ix])
            chi_high_mean = float(row[chi_high_mean_ix])
            chi[(g, ek)][iteration].append(
                (chi_diff, Y_diff, chi_low_mean, chi_high_mean)
            )

    # (iteration,
    # chi_diff_mean, Y_diff_mean, chi_low_mean_mean, chi_high_mean_mean,
    # chi_diff_err, Y_diff_err, chi_low_mean_err, chi_high_mean_err)
    chi_plot = {
        key: [[i] + list(np.average(np.transpose(tup), axis=1))
                  + list(np.std(np.transpose(tup), axis=1))
              for i, tup in iters.items()]
        for key, iters in chi.items()
    }
    chi_plot = sorted(chi_plot.items(),
                      key=lambda kv_pair: (kv_pair[0][0], kv_pair[0][1]))

    ###
    # plot chi_diff
    ###

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    handle = {}
    x = {}
    for i, ((g, ek), triples) in enumerate(chi_plot):
        x[i], yl, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[1], d[5]] for d in triples]
        )
        handle[(g, ek)] = plt.errorbar(
            x[i], yl, yerr=ye, label=label + str(g) + ', ' + str(ek)
        )
        # plot individual runs:
        chi_plot_individual = {
            i: np.transpose(tup)[0]
            for i, tup in chi[(g, ek)].items()
        }
        chi_diff_individual = np.transpose([
            list(c) for c in chi_plot_individual.values()
        ])
        for cdi in chi_diff_individual:
            plt.plot(list(chi_plot_individual.keys()), cdi,
                     color=handle[(g, ek)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
    ax.set_xticks(x[0])
    plt.xlabel(r'Number $k$ of iterations')
    plt.ylabel(r'Average $\left|\underline{\chi}^{(k)}_{p_1, p_2} - \overline{\chi}^{(k)}_{p_1, p_2} \right|$ over pairs $(p_1, p_2)$ along city-center boundary')
    plt.xlim(left=0)
    plt.ylim(*ylim_cc_diff)
    plt.grid(lw=0.2, axis='y')
    plt.legend(handles=list(handle.values()), loc='upper right')
    # plt.title('mean difference between upper and lower $\chi$ values')
    plt.savefig(img_path + f'chi_diff_{version}_{case}.pdf')
    # plt.savefig(img_path + f'benchmarks/chi_diff_{version}_{case}.pdf')
    plt.close()

    ###
    # plot chi
    ###

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    handle = {}
    x = {}
    for i, ((g, ek), triples) in enumerate(chi_plot):
        x[i], yl, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[3], d[7]] for d in triples]
        )
        p = plt.errorbar(x[i], yl, yerr=ye)
        x[i], yl, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[4], d[8]] for d in triples]
        )
        handle[(g, ek)] = plt.errorbar(
            x[i], yl, yerr=ye, label=label + str(g) + ', ' + str(ek),
            color=p[0].get_color()
        )
        # plot individual lower runs:
        chi_plot_individual = {
            i: np.transpose(tup)[2]
            for i, tup in chi[(g, ek)].items()
        }
        chi_diff_individual = np.transpose([
            list(c) for c in chi_plot_individual.values()
        ])
        for cdi in chi_diff_individual:
            plt.plot(list(chi_plot_individual.keys()), cdi,
                     color=handle[(g, ek)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
        # plot individual upper runs:
        chi_plot_individual = {
            i: np.transpose(tup)[3]
            for i, tup in chi[(g, ek)].items()
        }
        chi_diff_individual = np.transpose([
            list(c) for c in chi_plot_individual.values()
        ])
        for cdi in chi_diff_individual:
            plt.plot(list(chi_plot_individual.keys()), cdi,
                     color=handle[(g, ek)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
    ax.set_xticks(x[0])
    plt.xlabel(r'Number $k$ of iterations')
    plt.ylabel(r'Average of $\underline{\chi}^{(k)}_{p_1, p_2}$ and $\overline{\chi}^{(k)}_{p_1, p_2}$ over pairs $(p_1, p_2)$ along city-center boundary')
    plt.xlim(left=0)
    plt.ylim(*ylim_cc)
    plt.grid(lw=0.2, axis='y')
    plt.legend(handles=list(handle.values()), loc='upper right')
    # plt.title(r'mean upper and lower $\chi$ values')
    fig.subplots_adjust(left=0.14)
    plt.savefig(img_path + f'chi_mean_interval_{version}_{case}.pdf')
    # plt.savefig(img_path + f'benchmarks/chi_mean_interval_{version}_{case}.pdf')
    plt.close()

    ###
    # plot Y_diff
    ###

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # ax.set_yscale("log", nonposy='clip')
    ax.axvline(min_iter, linestyle=':', lw=1, color='k')  # horizontal lines
    handle = {}
    x = {}
    yh_dict = {}
    for i, ((g, ek), triples) in enumerate(chi_plot):
        x[i], yh, ye = np.transpose(
            [[d[0] + (-1)**i * i * group_offset, d[2], d[6]] for d in triples]
        )
        handle[(g, ek)] = plt.errorbar(
            x[i], yh, yerr=ye, label=f'{i+1}) {label}{g}, {ek}'
        )
        # plot individual runs:
        chi_plot_individual = {
            i: np.transpose(tup)[1]
            for i, tup in chi[(g, ek)].items()
        }
        chi_diff_individual = np.transpose([
            list(c) for c in chi_plot_individual.values()
        ])
        for cdi in chi_diff_individual:
            plt.plot(list(chi_plot_individual.keys()), cdi,
                     color=handle[(g, ek)][0].get_color(),
                     lw=individual_lw,
                     alpha=individual_alpha)
        yh_dict[(g, ek)] = yh

    ###
    # intermezzo: compute ratio between prediction differences
    ###

    print('Computing ratios between average difference between predicted '
          f'values along the boundary, averaged across all iterations '
          f'>= {min_iter}:')
    pos = []
    data = []
    for j, (key_nom, yh_nom) in enumerate(yh_dict.items()):
        for i, (key_denom, yh_denom) in enumerate(yh_dict.items()):
            if i >= j:
                continue
            pos.append(f'{i+1})/{j+1})')
            data.append(yh_nom[min_iter:] / yh_denom[min_iter:])
            ratio = np.mean(yh_nom[min_iter:] / yh_denom[min_iter:])
            ratio_std = np.std(yh_nom[min_iter:] / yh_denom[min_iter:])
            print(f'{key_nom} / {key_denom}: {ratio} +- {ratio_std}')

    # don't know how to dynamically generate latex format
    pos = [
        r'$\frac{2)}{1)}$',
        r'$\frac{3)}{1)}$',
        r'$\frac{3)}{2)}$',
        r'$\frac{4)}{1)}$',
        r'$\frac{4)}{2)}$',
        r'$\frac{4)}{3)}$',
        r'$\frac{5)}{1)}$',
        r'$\frac{5)}{2)}$',
        r'$\frac{5)}{3)}$',
        r'$\frac{5)}{4)}$',
        r'$\frac{6)}{1)}$',
        r'$\frac{6)}{2)}$',
        r'$\frac{6)}{3)}$',
        r'$\frac{6)}{4)}$',
        r'$\frac{6)}{5)}$',
    ]

    axins = inset_axes(ax, width='50%', height='31%')
    axins.violinplot(
        data,
        showmeans=True,
        showextrema=True,
        # showmedians=True
    )
    axins.set_ylim(bottom=0)
    plt.setp(axins, xticks=range(1, len(pos)+1), xticklabels=pos)
    plt.setp(axins.get_xticklabels(), fontsize=8)
    plt.setp(axins.get_yticklabels(), fontsize=8)
    # axins.set_ylabel(r'Ratios of average differences between different parameter runs', fontsize=8)
    axins.axhline(1, linestyle='--', lw=1, color='k')  # horizontal lines
    for i in range(1, len(yh_dict)-1):
        axins.axvline(i*(i+1)/2+0.5, lw=0.2, color='grey')  # vertical lines

    ax.set_xticks(x[0])
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'Average $\left|f^{(m)}_{p} - f^{(n)}_{p}\right|$ over points $p$ along city-center boundary')
    ax.set_xlim(left=0)
    ax.set_ylim(*ylim_y_diff)
    ax.grid(lw=0.2, axis='y')
    ax.legend(handles=list(handle.values()), loc='upper left')
    # ax.set_title('mean difference between upper and lower predictions')
    fig.subplots_adjust(left=0.15)
    plt.savefig(img_path + f'Y_diff_{version}_{case}.pdf')
    # plt.savefig(img_path + f'benchmarks/Y_diff_{version}_{case}.pdf')
    plt.close()

    return None


if __name__ == '__main__':
    plot_cost_and_cc()
