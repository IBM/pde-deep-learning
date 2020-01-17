from datetime import datetime
import numpy as np
import pymongo
import tensorflow as tf

import util.util_consistency_constraints as ucc
import util.util_ml_model as umm

""" Script for training the ML model on pre-processed data.

Description:
    This module contains the settings and main loop for training the 
    ML model

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

tf.logging.set_verbosity(tf.logging.INFO)


def get_parameters():
    """
    use 'mesh_size': 2 for synthetic example ('case': 'Demo')
    use 'mesh_size': 12 for Dublin example ('case': 'Dublin')
    select 'tiles': [i, j, k] for the tiles i, j, k that are part of the run
    The synthetic example has the two tiles 6 and 7. ('tiles': [6, 7])
    :return:
    """
    param = {
        # tags for data retrieval:
        'case': 'Demo',  # 'Dublin' or 'Demo'
        'mesh_size': 2,  # number of sub-domains in pre-processed collection
        'tiles': [6, 7],  # list of id's to be used in modeling
        # 'tiles': list(range(1, 12 + 1)),
        # model hyper-parameters
        'num_iterations': 20,
        'num_hidden_layers': 5,
        'num_nodes': 50,
        'num_epochs': 25,
        'batch_size': 128,
        'l2_reg_coefficient': 1e-4,  # weights are regularized with l2 norm
        'starter_learning_rate': 1e-3,  # higher values lead to initial divergence
        'decay_factor': 0.85,  # exponential decay
        'train_to_test_split': 0.9,  # train_%
        'add_previous_labels_to_input': False,  # True is not a feature
        # ToDo: allow True in update of consistency constraint data
        # consistency constraints
        'use_consistency_constraints': True,
        'cc_reg_coefficient': 10,  # lambda
        'kappa': 0.5,
        'epsilon': 0,
        'cc_update_version': 'version 3',
        # check util.util_consistency_constraints
        # saving of output
        'do_save_benchmark': True,
        'do_save_cc': True,
        'do_save_model': True,
        'do_save_estimates': False,
        'iterations_to_save_estimates': [1, 5, 10, 20],  # 1-based
        'do_print_status': True,
        # for reproducibility
        'random_seed': None
        # None or (int) < 2147483648. If None, it's taken at random
    }
    return param


def get_collections(port=27018):
    """
    get: collection of utility data
         collection to draw data from
         collection to write trained MLP output to
    :param port: of the mongoDB database
    :return: collections
    """
    client_internal = pymongo.MongoClient('localhost', port=port)
    util = client_internal.db_air_quality.util
    data = client_internal.db_air_quality.proc_estimates_NO2
    predictions = client_internal.db_air_quality.ml_estimates_cc_test
    collections = {'util': util,
                   'data': data,
                   'pred': predictions}
    return collections


def main():
    param = get_parameters()
    if param['random_seed'] is None:
        param['random_seed'] = np.random.randint(2147483647)
    np.random.seed(param['random_seed'])
    param['start_time'] = datetime.strftime(datetime.now(), '%y%m%d-%H%M%S')

    if param['do_print_status']:
        print(f'{datetime.strftime(datetime.now(), "%y.%m.%d-%H:%M:%S")}')
        print(f'Using pre-processed data for {param["case"]} '
              # f'with tag {param["tag"]}.'
              )
        print(f'Hidden layers: {param["num_hidden_layers"]}\t '
              f'nodes: {param["num_nodes"]}\t')
        print(f'learning rate: {param["starter_learning_rate"]}\t '
              f'decay rate: {param["decay_factor"]}')
        print(f'Lambda: {param["cc_reg_coefficient"]}\t '
              f'kappa: {param["kappa"]}\t '
              f'epsilon: {param["epsilon"]}\n')
        print('Loading data ...')
    collections = get_collections()
    if param['do_print_status']:
        print(f'Data is taken from {collections["data"].name}.')
        print(f'ML estimates are written to {collections["pred"].name}.')
    mesh = umm.get_mesh(collections['util'], **param)
    data = umm.get_data(collections['data'], mesh, **param)
    if param['do_print_status']:
        print('Data loaded successfully. '
              f'{datetime.strftime(datetime.now(), "%y.%m.%d-%H:%M:%S")}')
        all_data = np.concatenate(list(data['labels'].values()))
        print(f'Labels min: {np.min(all_data)}')
        print(f'Labels median: {np.median(all_data)}')
        print(f'Labels mean: {np.mean(all_data)}')
        print(f'Labels max: {np.max(all_data)}\n')

    normalisation_stats = collections['data'].find({
        'util.case': param['case']
    })[0]['util']['utils']

    # iterations are 1-based
    for iteration in range(1, param['num_iterations'] + 1):
        mlp_times = umm.run_recursion_cycle(data, mesh, iteration,
                                            collections['pred'],
                                            normalisation_stats, **param)
        if param['use_consistency_constraints']:
            ucc.update_consistency_constraints(data, mesh, iteration, **param)
        if param['do_print_status']:
            print(f'Average MLP run time for all data of one tile: '
                  f'{np.average(mlp_times):.6f}s')
            print(f'Standard Deviation: {np.std(mlp_times):.6f}s')
            print('')

    print(f'{datetime.strftime(datetime.now(), "%y.%m.%d-%H:%M:%S")}')

    return None


if __name__ == '__main__':
    for _ in range(6):
        main()
