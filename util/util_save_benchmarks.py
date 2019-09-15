from datetime import datetime
import numpy as np
import os
import time

""" Utility methods for saving benchmarks

Description:
    This module implements utility functions for saving the consistency 
    constraints and performance benchmarks in the given files below. 

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
    2019 - 08 - 30

"""

benchmark_path = "../output/benchmarks/"
benchmark_file_name = benchmark_path + "benchmark.xlsx"
benchmark_sheet_name = "with CC"
chi_file_name = benchmark_path + "chi.xlsx"
chi_sheet_name = "chi"


def save_consistency_constraints(cc_chi, y_diff, tile, neighbor, iteration,
                                 **kwargs):
    header = (
        'date\tseed\tmesh_size\t'
        + 'learning_rate\tbatch_size\tepochs\t'
        + 'n_layers\tn_nodes\t'
        + 'reg_coeff\tcc_coeff\tkappa\tepsilon\t'
        + 'tile\tneighbor\titer\t'
        + 'chi_l_min\tchi_l_med\tch_l_avg\tch_l_max\t'
        + 'chi_u_min\tchi_u_med\tch_u_avg\tch_u_max\t'
        + 'chi_avg_dist\tY_avg_dist\tversion'
        + '\n'
    )
    chi_save = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        kwargs['random_seed'],
        kwargs['mesh_size'],
        kwargs['starter_learning_rate'],
        kwargs['batch_size'],
        kwargs['num_epochs'],
        kwargs['num_hidden_layers'],
        kwargs['num_nodes'],
        kwargs['l2_reg_coefficient'],
        kwargs['cc_reg_coefficient'],
        kwargs['kappa'],
        kwargs['epsilon'],
        tile,
        neighbor,
        iteration,
        np.amin(cc_chi[tile][neighbor][0]),
        np.median(cc_chi[tile][neighbor][0]),
        np.mean(cc_chi[tile][neighbor][0]),
        np.amax(cc_chi[tile][neighbor][0]),
        np.amin(cc_chi[tile][neighbor][1]),
        np.median(cc_chi[tile][neighbor][1]),
        np.mean(cc_chi[tile][neighbor][1]),
        np.amax(cc_chi[tile][neighbor][1]),
        np.average(np.abs(
            cc_chi[tile][neighbor][1] - cc_chi[tile][neighbor][0]
        )),
        y_diff,
        kwargs['cc_update_version']
    ]
    line = '\t'.join([str(i) for i in chi_save]) + '\n'

    file_name = kwargs['start_time'] + '_' + kwargs['case'] + '.txt'
    file_path = '../output/cc/' + file_name
    file_exists = False
    if os.path.isfile(file_path):
        file_exists = True

    with open(file_path, 'a+') as f:
        if not file_exists:
            f.write(header)
        f.write(line)
        f.close()
    if kwargs['do_print_status']:
        print("Consistency constraints saved.")
    return None


def save_benchmarks(tile, iteration, num_instances, num_input, num_classes,
                    test_mse, test_mae, test_mape, test_smape,
                    training_start_time, mlp_times, **kwargs):
    if kwargs['do_save_benchmark']:
        header = (
            'date\tseed\tmesh_size\ttile\tn_instances\tn_input\tn_classes\t'
            + 'learning_rate\tdecay_factor\treg_coeff\tbatch_size\tepochs\t'
            + 'cc_coeff\tkappa\tn_layers\tn_nodes\titer\tMSE\tMAE\tMAPE\t'
            + 'sMAPE\ttraining_time\tML_execution_time'
            + '\n'
        )
        benchmark = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            kwargs['random_seed'],
            kwargs['mesh_size'],
            tile,
            num_instances,
            num_input,
            num_classes,
            kwargs['starter_learning_rate'],
            kwargs['decay_factor'],
            kwargs['l2_reg_coefficient'],
            kwargs['batch_size'],
            kwargs['num_epochs'],
            kwargs['cc_reg_coefficient'],
            kwargs['kappa'],
            kwargs['num_hidden_layers'],
            kwargs['num_nodes'],
            iteration,
            test_mse,
            test_mae,
            test_mape,
            test_smape,
            int(time.perf_counter() - training_start_time),
            np.average(mlp_times)
        ]
        line = '\t'.join([str(i) for i in benchmark]) + '\n'

        file_name = kwargs['start_time'] + '_' + kwargs['case'] + '.txt'
        file_path = '../output/benchmarks/' + file_name
        file_exists = False
        if os.path.isfile(file_path):
            file_exists = True

        with open(file_path, 'a+') as f:
            if not file_exists:
                f.write(header)
            f.write(line)
            f.close()
        if kwargs['do_print_status']:
            print("Benchmarks saved.")
    return None


def save_ml_estimates(estimates, inputs, iteration, collection_mlp_estim,
                      normalisation_stats, **kwargs):

    def inv_normalise(value, mean, std):
        return value * std + mean

    if (kwargs['do_save_estimates']
            and iteration in kwargs['iterations_to_save_estimates']):
        if kwargs['do_print_status']:
            print("Saving of MLP estimates ...")

        t_max = normalisation_stats['time']['max']
        t_min = normalisation_stats['time']['min']

        coord_mean = np.asarray(normalisation_stats['coord']['mean'])
        coord_std = np.asarray(normalisation_stats['coord']['std'])

        p_max = normalisation_stats['poll']['max']
        p_min = normalisation_stats['poll']['min']

        # if we try to save the whole estimates array, we may get an error:
        # pymongo.errors.DocumentTooLarge: BSON document too large
        save = []
        for i, xinput in enumerate(inputs):
            t = inv_normalise(xinput[0], t_min, t_max - t_min)
            for p_i, poll in enumerate(normalisation_stats['pollutants']):
                save.append({
                    'case': kwargs['case'],
                    'date': datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': t,
                    'pollutant': poll,
                    'coord': list(inv_normalise(xinput[-2:],
                                                coord_mean, coord_std)),
                    'value': inv_normalise(estimates[i][p_i],
                                           p_min[poll],
                                           p_max[poll] - p_min[poll]),
                    'iteration': iteration,
                    'settings': {
                        'gamma': kwargs['cc_reg_coefficient'],
                        'kappa': kwargs['kappa'],
                        'layers': kwargs['num_hidden_layers'],
                        'neurons': kwargs['num_nodes'],
                        'epochs': kwargs['num_epochs'],
                        'batch size': kwargs['batch_size'],
                        'learning rate': kwargs['starter_learning_rate'],
                        'comment': kwargs['cc_update_version'],
                        'seed': kwargs['random_seed']
                    }
                })
                # collect estimates in a batch and then write batch to database
                if len(save) < 100000:
                    continue
                collection_mlp_estim.insert_many(save)
                save = []
        # collect the last batch if not empty
        if len(save):
            collection_mlp_estim.insert_many(save)
    return None
