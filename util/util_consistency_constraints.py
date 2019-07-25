from collections import defaultdict
import numpy as np

import util.util_save_benchmarks as usb

""" Utility methods for using the consistency constraints

Description:
    This module implements utility functions for the construction of 
    consistency constraints, and the update steps.

-*- coding: utf-8 -*-

Legal:
    (C) Copyright IBM 2018.
    
    This code is licensed under the Apache License, Version 2.0. You may
    obtain a copy of this license in the LICENSE.txt file in the root directory
    of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
    
    Any modifications or derivative works of this code must retain this
    copyright notice, and modified files need to carry a notice indicating
    that they have been altered from the originals.

    IBM-Review-Requirement: Art30.3
    Please note that the following code was developed for the project VaVeL at
    IBM Research -- Ireland, funded by the European Union under the
    Horizon 2020 Program.
    The project started on December 1st, 2015 and was completed by December 1st,
    2018. Thus, in accordance with Article 30.3 of the Multi-Beneficiary General
    Model Grant Agreement of the Program, there are certain limitations in force 
    up to December 1st, 2022. For further details please contact Jakub Marecek
    (jakub.marecek@ie.ibm.com) or Gal Weiss (wgal@ie.ibm.com).

If you use the code, please cite our paper:
https://arxiv.org/abs/1810.09425

Authors: 
    Philipp Hähnel <phahnel@hsph.harvard.edu>

Last updated:
    2019 - 05 - 23

"""


def consistency_constraints_v1(y_low, y_high, epsilon):
    lower = np.minimum(y_low, y_high) + epsilon
    upper = np.maximum(y_low, y_high) - epsilon
    return lower, upper


def consistency_constraints_v2(y_low, y_high, chi, epsilon):
    lower = np.maximum(y_low, np.minimum(chi[0], y_high)) + epsilon
    upper = np.minimum(np.maximum(y_low, chi[1]), y_high) - epsilon
    return lower, upper


def consistency_constraints_v3(y_low, y_high, chi, chi_old, kappa, iteration):
    alpha = 1
    zeta = 1
    tau = kappa * 1 / (np.sqrt(iteration) + zeta)
    kap = kappa * np.sqrt(iteration) / (np.sqrt(iteration) + zeta)
    lower = alpha * chi[0] + tau * (y_low - chi[0]) + kap * (chi[0] - chi_old[0])
    upper = alpha * chi[1] + tau * (y_high - chi[1]) + kap * (chi[1] - chi_old[1])
    return lower, upper


def update_consistency_constraints(data, mesh, iteration, **kwargs):
    """
    :param data: output of get_data()
    :param mesh: output of get_mesh()
    :param iteration:
    :param kwargs:
    :return:
    """
    c_labels_new = data['cc_labels_new']
    cc_chi = data['cc_chi']
    cc_chi_old = data['cc_chi_old']

    batch_size = kwargs['batch_size']
    version = kwargs['cc_update_version']

    # update chi's
    cc_chi_new = defaultdict(dict)
    y_diff = defaultdict(dict)
    for tile in mesh['tiles']:
        n_labels = data['labels'][tile].shape[1]
        labels_min = np.amin([np.amin(tile_labels) for tile_labels in data['labels'].values()])
        labels_max = np.amax([np.amax(tile_labels) for tile_labels in data['labels'].values()])
        for neighbor in mesh['neighbors'][tile]:
            if not len(c_labels_new[tile][neighbor]):
                cc_chi_new[tile][neighbor] = [np.full([batch_size, n_labels], labels_min),
                                              np.full([batch_size, n_labels], labels_max)]
            else:
                y_low = np.minimum(c_labels_new[tile][neighbor], c_labels_new[neighbor][tile])
                y_high = np.maximum(c_labels_new[tile][neighbor], c_labels_new[neighbor][tile])
                # for the first iteration, those shapes are not the same
                if cc_chi[tile][neighbor][0].shape != y_low.shape:
                    cc_chi[tile][neighbor] = [np.full(y_low.shape, labels_min),
                                              np.full(y_high.shape, labels_max)]

                if version == 'version 1':
                    lower, upper = consistency_constraints_v1(y_low, y_high,
                                                              kwargs['epsilon'])
                elif version == 'version 2':
                    lower, upper = consistency_constraints_v2(y_low, y_high, cc_chi[tile][neighbor],
                                                              kwargs['epsilon'])
                else:  # 'version 3'
                    lower, upper = consistency_constraints_v3(y_low, y_high, cc_chi[tile][neighbor],
                                                              cc_chi_old[tile][neighbor],
                                                              kwargs['kappa'], iteration)
                cc_chi_new[tile][neighbor] = [lower, upper]
                y_diff[tile][neighbor] = np.average(np.abs(y_high - y_low))
    data['cc_chi_old'] = cc_chi
    data['cc_chi'] = cc_chi_new

    data['cc_input'] = data['cc_input_new']
    data['cc_input_new'] = defaultdict()
    data['cc_labels_new'] = defaultdict()
    for tile in mesh['tiles']:
        data['cc_input_new'][tile] = defaultdict(list)
        data['cc_labels_new'][tile] = defaultdict(list)
    data['receptor_pos'] = defaultdict(list)
    data['emitters']['tile'] = defaultdict(list)
    data['emitters']['ngbr'] = defaultdict(list)

    if kwargs['do_save_cc']:
        for tile in mesh['tiles']:
            for neighbor in mesh['neighbors'][tile]:
                if tile < neighbor:
                    usb.save_consistency_constraints(cc_chi_new, y_diff[tile][neighbor],
                                                     tile=tile, neighbor=neighbor,
                                                     iteration=iteration + 1, **kwargs)

    return None


def get_boundary_receptor_emission_data(data, tile, neighbor, boundary, num_samples):
    """
        Used in update_consistency_data() below
    """
    c_emitters = data['emitters']
    if len(c_emitters['tile'][boundary]):
        c_input_tile = c_emitters['ngbr'][boundary]
        c_input_ngbr = c_emitters['tile'][boundary]
    else:
        timestamps = []
        for instance in data['input'][tile]:
            # skip known timestamps
            if instance[0] in timestamps:
                continue
            for n_instance in data['input'][neighbor]:
                # select if same timestamp as neighbor
                if instance[0] == n_instance[0]:
                    c_emitters['tile'][boundary].append(instance)
                    c_emitters['ngbr'][boundary].append(n_instance)
                    break
            timestamps.append(instance[0])
        max_len = int(len(c_emitters['tile'][boundary]) / num_samples) * num_samples

        c_input_tile = c_emitters['tile'][boundary][:max_len]
        c_input_ngbr = c_emitters['ngbr'][boundary][:max_len]
    return c_input_tile, c_input_ngbr


def move_receptor_positions_to_boundary(c_input_tile, mesh, data, tile, neighbor, boundary, num_samples, **kwargs):
    """
        Used in update_consistency_data() below
    """
    # choose consistency receptors at boundary
    c_receptor_pos = data['receptor_pos']
    n_input = data['input'][tile].shape[1]
    n_labels = data['labels'][tile].shape[1]
    # 1 = len(timestamp); 4 = len(weather_data); 5 = len(traffic_source)
    emitter_len = 1 + 4 + 5 * mesh['max_links']
    if kwargs['add_previous_labels_to_input']:
        num_c_receptors = int((n_input - n_labels - emitter_len) / 2)  # 2 coords per receptor
    else:
        num_c_receptors = int((n_input - emitter_len) / 2)  # 2 coords per receptor
    border = mesh['intersections'][tile][neighbor]
    if not len(c_receptor_pos[boundary]):
        # generate 3*num_c_receptors possible positions and then choose num_c_receptors many of them
        receptor_pos = np.transpose(list(map(lambda x1, x2: np.linspace(x1, x2, 3 * num_c_receptors),
                                             border[0], border[1])))
        selection = np.random.choice(receptor_pos.shape[0], size=num_c_receptors, replace=False)
        c_receptor_pos[boundary] = receptor_pos[selection, :]
        # normalise as in pre-processing (all are np arrays, so operations are element-wise)
        c_receptor_pos[boundary] = [(pos - mesh['coord_mean']) / mesh['coord_std']
                                    for pos in c_receptor_pos[boundary]]
        # flatten list
        c_receptor_pos[boundary] = [coordinate for pos in c_receptor_pos[boundary]
                                    for coordinate in pos]
    # form new consistency data
    pre_consistency_data = [list(emitters[:emitter_len]) + list(c_receptor_pos[boundary])
                            for emitters in c_input_tile]
    # select num_samples many samples
    pre_consistency_data = pre_consistency_data[:int(len(pre_consistency_data) / num_samples) * num_samples]
    return pre_consistency_data
