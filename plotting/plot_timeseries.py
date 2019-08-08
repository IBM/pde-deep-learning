from collections import defaultdict
import datetime

#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt

import numpy as np
import pymongo

import util.util_db_access as uda


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
    2019 - 08 - 02

"""


def get_parameters():
    """
    :return:
    """
    #  time slice (2017-07-01 01:00:00 to 2018-05-02 14:00:00)
    param = {
        'date_start': datetime.datetime(2017, 11, 19, 0),
        'period': datetime.timedelta(days=3),
        'receptor_coord': [53.34421064496467, -6.26476486860426],
        'station_name': 'Winetavern St Civic Offices',
        # is appended to file name:
        'pollutants': ['NO2']
    }
    return param


def plot_timeseries(
        date_start, period, receptor_coord, station_name, pollutants,
        # gamma=1, kappa=0.5, iter=2, seed=1753245344
        **kwargs
):

    timestamp_start = date_start.timestamp()
    # there is a mismatch between dates and labels
    date_print = date_start - datetime.timedelta(hours=16)
    date_end = date_start + period
    site = tuple(receptor_coord)

    print('Getting station measurement data ...')
    station_measurements = uda.get_station_measurements(
        collection_measurements=collection_measurement,
        date_start=date_start,
        date_end=date_end,
        station_name=station_name
    )
    station_time_series = defaultdict(list)
    polls = []
    for timestamp, pollution_values in station_measurements.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            if pollutant not in polls:
                polls.append(pollutant)
            station_time_series[pollutant].append((t, value))

    print('Getting background pollution data ...')
    background_pollution = uda.get_background_pollution(
        collection_measurements=collection_measurement,
        date_start=date_start,
        date_end=date_end
    )
    background_time_series = defaultdict(list)
    for timestamp, pollution_values in background_pollution.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            background_time_series[pollutant].append((t, value))

    print('Getting Caline estimates ...')
    caline_estimates = uda.get_caline_estimates_for_receptor(
        collection_caline_estimates=collection_caline_estimates,
        date_start=date_start,
        date_end=date_end,
        receptor_coord=receptor_coord
    )
    caline_time_series = defaultdict(list)
    timestamps = []
    for timestamp, pollution_values in caline_estimates.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            if t not in timestamps:
                timestamps.append(t)
            caline_time_series[pollutant].append((t, value))

    print('Getting MLP estimates ...')
    ml_filter = {
        # 'settings.gamma': gamma,
        # 'settings.kappa': kappa,
        # 'settings.iteration': iter
    }
    ml_estimates = uda.get_ml_estimates_for_receptor(
        collection_ml_estimates=collection_estim,
        collection_util=collection_util,
        date_start=date_start,
        date_end=date_end,
        receptor_coord=receptor_coord,
        **ml_filter
    )
    ml_time_series = defaultdict(list)
    for timestamp, pollution_values in ml_estimates.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            ml_time_series[pollutant].append((t, value))

    # mlp_labels = np.asarray(list(reversed([label[(index-1)*3] for label in entry['labels']])))

    timestamps.sort()

    print('Plotting time series ...')
    for p, poll in enumerate(pollutants):
        fig = plt.figure(figsize=(10, 10))
        sub = fig.add_subplot(1, 1, 1)

        [station_x, station_y] = np.transpose(station_time_series[poll])
        station, = plt.plot(station_x, station_y, 'b.',
                            label='station measurements')
        handles = [station]

        [bgrd_x, bgrd_y] = np.transpose(background_time_series[poll])

        if poll in caline_time_series:
            [x, y] = np.transpose(caline_time_series[poll])
            y += bgrd_y
            caline, = plt.plot(x, y, 'r+', label='Caline estimates')
            handles.append(caline)

        if poll in ml_time_series:
            [ml_x, ml_y] = np.transpose(ml_time_series[poll])
            ml_y += bgrd_y
            mlp, = plt.plot(ml_x, ml_y, 'gx', label='MLP estimates')
            handles.append(mlp)

        loc_x = list(np.linspace(x.min(), x.max(), 4))
        label_x = list(range(0, 4*24, 24))
        plt.xticks(loc_x, label_x)
        sub.set_xticks(x, minor=True)
        label_y = list(range(0, 81, 10))
        plt.yticks(label_y, label_y)
        sub.grid(which='major')

        plt.xlabel('Hours from ' + date_print.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel(poll + ' concentration [micrograms/m3]')
        plt.title('Pollution Data in Dublin City Center '
                  '(near ' + str(receptor_coord) + ')')
        plt.legend(handles=handles)
        plot_name = (
            img_path + 'timeseries/timeseries_' + poll + '_'
            + str(site) + '_' + date_start.strftime('%Y-%m-%d_%H-%M-%S')
        )
        plt.savefig(plot_name + '.pdf')
        # plt.savefig(plot_name + '.png')
        plt.close()

    return None


if __name__ == '__main__':

    benchmark_path = '../output/benchmarks/'
    img_path = '../output/img/'

    """ connect to internal Mongo database """
    client = pymongo.MongoClient('localhost', 27018)
    # 2017-07-01 01:00:00 to 2018-05-02 14:00:00
    collection_measurement = client.db_air_quality.pollution_measurements
    collection_caline_estimates = client.db_air_quality.caline_estimates
    collection_estim = client.db_air_quality.ml_estimates
    collection_util = client.db_air_quality.util

    plot_timeseries(**get_parameters())
