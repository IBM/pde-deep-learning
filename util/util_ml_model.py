from collections import defaultdict
from operator import add
import numpy as np
import pymongo
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

import util.util_db_access as uda
import util.util_consistency_constraints as ucc
import util.util_save_benchmarks as usb

""" Utility methods for running the ML model

Description:
    This module implements utility functions for the training of the ML 
    model. The model_path sets where the intermediately trained models 
    are saved and restored from.

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
    Fearghal O'Donncha <feardonn@ie.ibm.com>

Last updated:
    2019 - 08 - 30

"""

model_path = "../output/models/"


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
    data = client_internal.db_air_quality.proc_estimates
    predictions = client_internal.db_air_quality.ml_estimates
    collections = {'util': util,
                   'data': data,
                   'pred': predictions}
    return collections


def get_mesh(collection_util, **kwargs):
    """
    :param collection_util: utilities collection
    :return: utility dict containing reduced neighbor and intersection
    information
    """
    utilities = uda.get_utilities_from_collection(collection_util,
                                                  case=kwargs['case'])
    max_links = max([
        len(domain['links']) for domain in utilities['domain_dict'].values()
    ])
    # important to select only those tiles and neighbors that are there!
    neighbors_select = {k: [n for n in v if n in kwargs['tiles']]
                        for k, v in utilities['domain_neighbors'].items()
                        if k in kwargs['tiles']}
    intersections_select = {k: {kv: vv for kv, vv in v.items()
                                if kv in kwargs['tiles']}
                            for k, v in utilities['intersections'].items()
                            if k in kwargs['tiles']}
    bounding_box = np.transpose(list(utilities['bounding_box'].values()))

    mesh = {'max_links': max_links,
            'neighbors': neighbors_select,
            'intersections': intersections_select,
            'coord_mean': np.mean(bounding_box, 1),
            'coord_std': np.std(bounding_box, 1),
            'tiles': kwargs['tiles'],
            'size': kwargs['mesh_size']
            }
    return mesh


def get_data(collection_data, mesh, **kwargs):
    """
    :param collection_data: pre-processed data
    :param mesh: output of get_mesh()
    :param kwargs:
    :return: data
    """
    inputs = defaultdict()
    labels = defaultdict()
    c_input_new = defaultdict()
    c_labels_new = defaultdict()
    cc_chi = defaultdict()
    for tile in mesh['tiles']:
        coll = [e for e in collection_data.find({'mesh_size': mesh['size'],
                                                 'sub_domain': tile})]
        x = [entry['input'] for entry in coll]
        y = [entry['labels'] for entry in coll]
        if kwargs['add_previous_labels_to_input']:
            # add labels of previous timestep to input
            # (first input entry is removed)
            x = list(map(add, x[1:], y[:-1]))
            del y[0]
        inputs[tile] = np.asarray(x)
        labels[tile] = np.asarray(y)
        c_input_new[tile] = defaultdict(list)
        c_labels_new[tile] = defaultdict(list)
        cc_chi[tile] = defaultdict()
    # Y_min = np.amin([np.amin(Y_tile) for Y_tile in labels.values()])
    # Y_max = np.amax([np.amax(Y_tile) for Y_tile in labels.values()])
    # Y_avg = np.average([np.average(Y_tile) for Y_tile in labels.values()])
    # print("Y_min: %.3f\t Y_avg: %.3f\t Y_max: %.3f\n"
    #       % (Y_min, Y_avg, Y_max))

    data = {'input': inputs,  # {tile: input list}
            'labels': labels,  # {tile: labels list}
            'cc_input': c_input_new,
            'cc_input_new': c_input_new,
            'cc_labels_new': c_labels_new,
            'cc_chi': cc_chi,  # {tile: {neighbor: receptor list}}
            'cc_chi_old': cc_chi,
            'receptor_pos': defaultdict(list),
            'emitters': {'tile': defaultdict(list),
                         'ngbr': defaultdict(list)}
            }
    return data


def multilayer_perceptron(input_data, weights, biases, num_hidden_layers=4):
    """
        A multilayer perceptron model
        input and hidden layers have ReLU activation function
        output layer has linear activation function
    :param input_data:
    :param weights:
    :param biases:
    :param num_hidden_layers:
    :return:
    """
    def dense(inp, w, b):
        return tf.add(tf.matmul(inp, w), b)

    hidden = dense(input_data, weights['in'], biases['in'])
    hidden = tf.nn.relu(hidden)

    for i in range(num_hidden_layers):
        hkey = 'h' + str(i + 1)
        bkey = 'b' + str(i + 1)
        hidden = dense(hidden, weights[hkey], biases[bkey])
        hidden = tf.nn.relu(hidden)

    hidden = dense(hidden, weights['out'], biases['out'])
    # out_layer = tf.nn.relu(hidden)
    out_layer = hidden
    return out_layer


def get_learning_rate(epoch, total_batch, **kwargs):
    """ modularized for more fine tuning options """
    decay_steps = 5000
    lr = (kwargs['starter_learning_rate']
          * kwargs['decay_factor'] ** (epoch * total_batch / decay_steps)
    )
    return lr


def run_recursion_cycle(data, mesh, iteration, collection_mlp_estim,
                        normalisation_stats, **kwargs):
    """
    :param data:
    :param mesh:
    :param iteration: 1-based
    :param collection_mlp_estim:
    :param normalisation_stats:
    :param kwargs: params
    :return:
    """

    mlp_times = []
    for tile_num, tile in enumerate(kwargs['tiles']):
        if kwargs['do_print_status']:
            print(f'Training for tile {tile} ({tile_num + 1}/{mesh["size"]}) '
                  f'at iteration {iteration}/{kwargs["num_iterations"]}')
        training_start_time = time.perf_counter()

        ############################
        # Normalize and initialize # __________________________________________
        ############################

        labels = np.asarray(data['labels'][tile])
        num_classes = labels.shape[1]  # Number of classes (output size)

        scaler = preprocessing.StandardScaler()
        scaler_wrap = preprocessing.StandardScaler()
        do_normalize = False
        if kwargs['add_previous_labels_to_input']:
            scaled_inputs = np.asarray(data['input'][tile][:, :-num_classes])
            scaled_labels = np.asarray(data['input'][tile][:, -num_classes:])
            if do_normalize:
                scaled_inputs = scaler.fit_transform(scaled_inputs)
                scaled_labels = scaler_wrap.fit_transform(scaled_labels)
            inputs = np.concatenate([scaled_inputs, scaled_labels], axis=1)
        else:
            # normalize the data
            inputs = np.asarray(data['input'][tile])
            if do_normalize:
                inputs = scaler.fit_transform(inputs)

        num_neighbors = len(mesh['neighbors'][tile])
        num_instances = inputs.shape[0]  # Number of instances
        num_input = inputs.shape[1]  # Input size

        if kwargs['do_print_status']:
            print(f'Number of inputs \t {num_input}')
            print(f'Number of labels \t\t {num_classes}')
            print(f'Number of instances \t {num_instances}')

        labels_min = np.amin(
            [np.amin(Y_tile) for Y_tile in data['labels'].values()]
        )
        labels_max = np.amax(
            [np.amax(Y_tile) for Y_tile in data['labels'].values()]
        )

        cc_input_train = {}
        if kwargs['use_consistency_constraints']:
            for neighbor in mesh['neighbors'][tile]:
                if iteration == 1:
                    cc_input_train[neighbor] \
                        = data['cc_input'][tile][neighbor]
                    if do_normalize:
                        cc_input_train[neighbor] = scaler.transform(
                            cc_input_train[neighbor]
                        )
                    # chi's are set at the end of each iteration
                else:
                    cc_input_train[neighbor] = np.zeros(
                        [kwargs['batch_size'], num_input]
                    )
                    data['cc_chi'][tile][neighbor] = [
                        np.full([kwargs['batch_size'], num_classes],
                                labels_min),
                        np.full([kwargs['batch_size'], num_classes],
                                labels_max)]
                    if kwargs['do_save_cc']:
                        if tile < neighbor:
                            usb.save_consistency_constraints(
                                data['cc_chi'],
                                labels_max - labels_min,
                                tile, neighbor,
                                iteration,
                                **kwargs
                            )
        # Train-test split
        input_train, input_test, labels_train, labels_test = train_test_split(
            inputs,
            labels,
            test_size=1 - kwargs['train_to_test_split'],
            random_state=100
        )

        #############
        # Get Graph # _________________________________________________________
        #############

        n_hidden = [kwargs['num_nodes']] * (1 + kwargs['num_hidden_layers'])

        with tf.name_scope('tile' + str(tile)):
            # tf Graph input
            tf_data = tf.placeholder(tf.float64, [None, num_input])
            tf_labels = tf.placeholder(tf.float64, [None, num_classes])

            # Initialise as random
            # Store layers weight & bias
            weights = {
                'in': tf.Variable(
                    tf.random_normal([num_input, n_hidden[0]],
                                     dtype=tf.float64)
                ),
                'out': tf.Variable(
                    tf.random_normal([n_hidden[-1], num_classes],
                                     dtype=tf.float64)
                )
            }
            biases = {
                'in': tf.Variable(
                    tf.random_normal([n_hidden[0]], dtype=tf.float64)
                ),
                'out': tf.Variable(
                    tf.random_normal([num_classes], dtype=tf.float64)
                )
            }
            for i in range(kwargs['num_hidden_layers']):
                hkey = 'h' + str(i + 1)
                bkey = 'b' + str(i + 1)
                weights[hkey] = tf.Variable(
                    tf.random_normal([n_hidden[i], n_hidden[i + 1]],
                                     dtype=tf.float64)
                )
                biases[bkey] = tf.Variable(
                    tf.random_normal([n_hidden[i + 1]], dtype=tf.float64)
                )

            tf_cc_input = tf.placeholder(tf.float64)
            chi = tf.placeholder(tf.float64)

        ###################
        # Construct model # ___________________________________________________
        ###################

        # predictions
        predictions = multilayer_perceptron(tf_data, weights, biases,
                                            kwargs['num_hidden_layers'])

        # regularizer for the weights
        regularizer = 0
        for weight in weights.values():
            regularizer += tf.nn.l2_loss(weight)

        # Define consistency constraints
        cc_cost = 0
        # dicts can't be fed, only arrays, which are fed in same order
        # of the neighbors
        if kwargs['use_consistency_constraints']:
            for n in range(num_neighbors):
                cc = multilayer_perceptron(tf_cc_input[n], weights, biases,
                                           kwargs['num_hidden_layers'])
                zero = tf.fill(tf.shape(cc), tf.constant(0, dtype=tf.float64))
                cc_cost += tf.reduce_mean(
                    tf.abs(tf.maximum(zero, cc - chi[n][0])
                           + tf.maximum(zero, chi[n][1] - cc))
                )

        # construct cost, metrics, and optimizer
        cost = (
            tf.reduce_mean(tf.square(predictions - tf_labels))
            + kwargs['l2_reg_coefficient'] * regularizer
            + kwargs['cc_reg_coefficient'] * cc_cost
        )
        mse = tf.reduce_mean(tf.square(predictions - tf_labels))
        mae = tf.reduce_mean(tf.abs(predictions - tf_labels))
        mape = tf.reduce_mean(tf.abs(predictions - tf_labels)
                              / tf.abs(tf_labels))
        smape = tf.reduce_mean(
            tf.abs((predictions - tf_labels)
                   / (tf.abs(predictions) + tf.abs(tf_labels)))
        )  # based on 100%

        global_step = tf.Variable(0, trainable=False)
        # learning rate is defined based on epoch
        learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(cost, global_step=global_step)

        #########
        # Saver # _____________________________________________________________
        #########

        saver = tf.train.Saver({**weights,
                                **biases,
                                'global_step': global_step})
        save_path = (model_path + "model"
                     + "-" + str(kwargs['num_hidden_layers'])
                     + "-" + str(kwargs['num_nodes'])
                     + "-" + str(kwargs['cc_reg_coefficient'])
                     + "-" + str(kwargs['kappa'])
                     + "-" + str(tile)
                     + "-" + str(kwargs['random_seed'])
                     )

        ###############
        # Train model # _______________________________________________________
        ###############

        if kwargs['do_print_status']:
            print('Training model:')

        training_set_size = int(num_instances * kwargs['train_to_test_split'])
        total_batch = int(input_train.shape[0] / kwargs['batch_size'])

        # with tf.Session(graph=graphs[tile]) as sess:
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if iteration > 1 and kwargs['do_save_model']:
                saver.restore(sess, save_path)
                print("Model restored.")

            # Start input enqueue threads.
            coord = tf.train.Coordinator()

            # Training loop
            t = time.perf_counter()
            for epoch in range(kwargs['num_epochs']):
                avg_cost = 0.
                # Loop over all batches
                for i in range(total_batch):
                    randidx = np.random.randint(training_set_size,
                                                size=kwargs['batch_size'])
                    # normal data
                    batch_xs = input_train[randidx, :]
                    batch_ys = labels_train[randidx, :]
                    # data for consistency constraints
                    cc_batch_xs = []
                    cc_batch_chi = []
                    if kwargs['use_consistency_constraints']:
                        for neighbor in mesh['neighbors'][tile]:
                            max_input_len = max(1,
                                                len(cc_input_train[neighbor]))
                            randidx_c = np.random.randint(
                                max_input_len, size=kwargs['batch_size']
                            )
                            cc_batch_xs.append(
                                cc_input_train[neighbor][randidx_c, :]
                            )
                            cc_batch_chi.append([
                                data['cc_chi'][tile][neighbor][0][randidx_c, :],
                                data['cc_chi'][tile][neighbor][1][randidx_c, :]
                            ])
                    # ToDo: allow for previous time stamp labels to be
                    #       wrapped to input
                    # -> what are the previous time stamp labels for
                    # the new boundary receptors?

                    lr = get_learning_rate(epoch, total_batch, **kwargs)
                    sess.run(optimizer,
                             feed_dict={tf_data: batch_xs, tf_labels: batch_ys,
                                        tf_cc_input: cc_batch_xs,
                                        chi: cc_batch_chi,
                                        learning_rate: lr})
                    # Compute average loss
                    avg_cost += sess.run(
                        cost,
                        feed_dict={tf_data: batch_xs,
                                   tf_labels: batch_ys,
                                   tf_cc_input: cc_batch_xs,
                                   chi: cc_batch_chi}
                    ) / total_batch

                # Display logs per epoch step
                if epoch % 1 == 0:
                    if kwargs['do_print_status']:
                        print(f'Epoch: {epoch}/{kwargs["num_epochs"]} cost: '
                              f'{avg_cost:.3f} ({time.perf_counter() - t:.2f}'
                              f's)')
                    t = time.perf_counter()

            if kwargs['do_print_status']:
                print("End of training.")

            ##############
            # Test model # ____________________________________________________
            ##############

            if kwargs['do_print_status']:
                print("Testing ...")

            test_acc = sess.run(mse, feed_dict={tf_data: input_test,
                                                tf_labels: labels_test})
            test_mae = sess.run(mae, feed_dict={tf_data: input_test,
                                                tf_labels: labels_test})
            test_mape = sess.run(mape, feed_dict={tf_data: input_test,
                                                  tf_labels: labels_test})
            test_smape = sess.run(smape, feed_dict={tf_data: input_test,
                                                    tf_labels: labels_test})
            if kwargs['do_print_status']:
                print(f'MSE: {test_acc:.3f}')
                print(f'MAE: {test_mae:.3f}')
                print(f'MAPE: {test_mape:.3f}')
                print(f'sMAPE: {test_smape:.3f}')

            ##################
            # Save estimates # ________________________________________________
            ##################

            mlp_run_time = time.perf_counter()
            # inputs is scaled data['input'][tile]
            estimates = sess.run(predictions, feed_dict={tf_data: inputs})
            mlp_times.append(time.perf_counter() - mlp_run_time)

            usb.save_ml_estimates(estimates, data['input'][tile], iteration,
                                  collection_mlp_estim, normalisation_stats,
                                  **kwargs)

            ###########################
            # Update consistency data # _______________________________________
            ###########################
            # -----------------------------------------------------------------
            # generate predictions at boundary and add them to respective
            # bordering tile input data.

            if kwargs['use_consistency_constraints']:
                if kwargs['do_print_status']:
                    print("Adding consistency data ...")

                c_input_new = data['cc_input_new']
                c_labels_new = data['cc_labels_new']
                # 1 == len(timestamp)
                # 4 == len(weather_data)
                # 5 == len(traffic_source)
                emitter_len = 1 + 4 + 5 * mesh['max_links']

                for neighbor in mesh['neighbors'][tile]:
                    boundary = tuple(sorted([tile, neighbor]))
                    # choose a number of inputs for which to use the
                    # emission data for boundary receptors
                    # number of samples needs to be multiple of batch_size
                    c_input_tile, \
                        c_input_ngbr = ucc.get_boundary_receptor_emission_data(
                            data, tile, neighbor, boundary,
                            num_samples=kwargs['batch_size']
                        )
                    # replace receptor positions of the selection with
                    # positions at boundary
                    pre_consistency_data \
                        = ucc.move_receptor_positions_to_boundary(
                            c_input_tile, mesh, data, tile, neighbor, boundary,
                            num_samples=kwargs['batch_size'], **kwargs
                        )
                    # Get boundary predictions:
                    # use trained model to predict labels at boundary
                    # transform coordinate choices to normalised input
                    # data for computing labels
                    if do_normalize:
                        pre_consistency_data = scaler.transform(
                            pre_consistency_data
                        )
                    # compute labels
                    batch_ranges = range(0, len(pre_consistency_data) + 1,
                                         kwargs['batch_size'])
                    c_labels = []
                    for i, j in zip(batch_ranges, batch_ranges[1:]):
                        pred = sess.run(
                            predictions,
                            feed_dict={tf_data: pre_consistency_data[i:j]}
                        )
                        c_labels.append(pred)
                    consistency_labels = [
                        labels
                        for label_array in c_labels
                        for labels in label_array
                    ]
                    if do_normalize:
                        # transform back the data
                        pre_consistency_data = scaler.inverse_transform(
                            pre_consistency_data)
                    consistency_data = pre_consistency_data
                    # replace wind and traffic data with data from
                    # neighboring tile
                    consistency_data[:emitter_len] = c_input_ngbr[:emitter_len]

                    c_input_new[neighbor][tile] = (
                        np.concatenate((c_input_new[neighbor][tile],
                                        consistency_data))
                        if len(c_input_new[neighbor][tile])
                        else consistency_data
                    )
                    c_labels_new[neighbor][tile] = (
                        np.concatenate((c_labels_new[neighbor][tile],
                                        consistency_labels))
                        if len(c_labels_new[neighbor][tile])
                        else consistency_labels
                    )

                data['cc_input_new'] = c_input_new
                data['cc_labels_new'] = c_labels_new

            #############################
            # Save benchmarks and model # _____________________________________
            #############################

            usb.save_benchmarks(tile, iteration, num_instances, num_input,
                                num_classes,
                                test_acc, test_mae, test_mape, test_smape,
                                training_start_time, mlp_times, **kwargs)

            if kwargs['do_save_model']:
                save = saver.save(sess, save_path)
                if kwargs['do_print_status']:
                    print("Model saved as %s" % save)

            ##################
            # Halt Processes # ________________________________________________
            ##################

            coord.request_stop()

        if kwargs['do_print_status']:
            print('')

    return mlp_times
