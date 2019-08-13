import datetime
import numpy as np
import openpyxl
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
    2019 - 08 - 01

"""

benchmark_path = "../output/benchmarks/"
benchmark_file_name = benchmark_path + "benchmark.xlsx"
benchmark_sheet_name = "with CC"
chi_file_name = benchmark_path + "chi.xlsx"
chi_sheet_name = "chi"


def save_consistency_constraints(cc_chi, y_diff, tile, neighbor, iteration,
                                 **kwargs):
    chi_save = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    chi_file = openpyxl.load_workbook(chi_file_name)
    chi_sheet = chi_file[chi_sheet_name]
    chi_sheet.append(chi_save)
    chi_file.save(chi_file_name)
    return None


def save_benchmarks(tile, iteration, num_instances, num_input, num_classes,
                    test_mse, test_mae, test_smape, test_mase,
                    training_start_time, mlp_times, **kwargs):
    if kwargs['do_save_benchmark']:
        benchmark = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            kwargs['random_seed'],
            kwargs['mesh_size'],
            tile,
            num_instances,
            num_input,
            num_classes,
            kwargs['starter_learning_rate'],
            kwargs['l2_reg_coefficient'],
            kwargs['batch_size'],
            kwargs['num_epochs'],
            kwargs['cc_reg_coefficient'],
            kwargs['kappa'],
            kwargs['num_hidden_layers'],
            kwargs['num_nodes'],
            iteration + 1,
            test_mse,
            test_mae,
            test_smape,
            test_mase,
            int(time.perf_counter() - training_start_time),
            np.average(mlp_times)
        ]
        benchmark_file = openpyxl.load_workbook(benchmark_file_name)
        benchmark_sheet = benchmark_file[benchmark_sheet_name]
        benchmark_sheet.append(benchmark)
        benchmark_file.save(benchmark_file_name)
        if kwargs['do_print_status']:
            print("Benchmarks saved.")
    return None


def save_ml_estimates(estimates, inputs, iteration, tile, collection_mlp_estim,
                      **kwargs):
    if (kwargs['do_save_estimates']
            and iteration in kwargs['iterations_to_save_estimates']):
        if kwargs['do_print_status']:
            print("Saving of MLP estimates ...")
        estimates = [list(estimate) for estimate in estimates]
        # if we try to save the whole estimates array, we may get an error:
        # pymongo.errors.DocumentTooLarge: BSON document too large
        save = []
        for i, xinput in enumerate(inputs):
            save.append({
                'tile': tile,
                'input': xinput,
                'labels': estimates[i],
                'settings': {
                    'gamma': kwargs['cc_reg_coefficient'],
                    'kappa': kwargs['kappa'],
                    'iteration': iteration,
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
