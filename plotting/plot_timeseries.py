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
import util.util_measurements as um


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
    2019 - 08 - 30

"""


def get_parameters(receptor_coord=(53.34421064496467, -6.26476486860426)):
    """
    :return:
    """
    #  time slice (2017-07-01 01:00:00 to 2018-05-02 14:00:00)
    param = {
        'date_start': datetime.datetime(2017, 10, 22, 0),
        'period': datetime.timedelta(days=3),
        'receptor_coord': list(receptor_coord),
        'station_name': 'Winetavern St Civic Offices',
        'iteration': 10,
        # is appended to file name:
        'pollutants': ['NO2']
    }
    return param


def plot_timeseries(
        date_start, period, receptor_coord, station_name, pollutants,
        # gamma=1, kappa=0.5, seed=1753245344
        **kwargs
):

    timestamp_start = date_start.timestamp()
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
    caline_estimates = uda.get_estimates_for_receptor(
        collection_estimates=collection_caline_estimates,
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
        'iteration': kwargs['iteration']
        # 'settings.gamma': gamma,
        # 'settings.kappa': kappa
    }
    ml_estimates = uda.get_estimates_for_receptor(
        collection_estimates=collection_estim,
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

    timestamps.sort()

    print('Plotting time series ...')
    for p, poll in enumerate(pollutants):

        if poll not in caline_time_series or poll not in ml_time_series:
            continue

        [bgrd_x, bgrd_y] = np.transpose(background_time_series[poll])
        [station_x, station_y] = np.transpose(station_time_series[poll])
        [x, y] = np.transpose(caline_time_series[poll])
        [ml_x, ml_y] = np.transpose(ml_time_series[poll])
        y += bgrd_y
        ml_y += bgrd_y

        min_y = int(np.min([np.min(y), np.min(ml_y), np.min(station_y)]))
        max_y = int(np.max([np.max(y), np.max(ml_y), np.max(station_y)]))

        # plot timeseries

        fig = plt.figure(figsize=(10, 10))
        st = plt.suptitle('Pollution Data in Dublin City Center\n'
                          f'at receptor location {str(receptor_coord)},'
                          '\n'
                          'compared to measurement station '
                          f'at {um.get_stations()[station_name]}')

        sub1 = fig.add_subplot(2, 1, 1)
        station, = plt.plot(station_x, station_y, 'b.',
                            label='Station measurements')
        handles = [station]
        if poll in caline_time_series:
            caline, = plt.plot(x, y, 'r+', label='Caline estimates')
            handles.append(caline)
        if poll in ml_time_series:
            label = f'ML estimates ({kwargs["iteration"]}. iter)'
            mlp, = plt.plot(ml_x, ml_y, 'gx', label=label)
            handles.append(mlp)

        loc_x = list(np.linspace(x.min(), x.max(), 4))
        label_x = list(range(0, 4*24, 24))
        label_y = list(range(10 * int((min_y - 1) / 10), max_y + 1, 10))
        plt.xticks(loc_x, label_x)
        plt.yticks(label_y, label_y)
        sub1.set_xticks(x, minor=True)
        sub1.grid(which='major')

        plt.xlabel('Hours from ' + date_start.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel(poll + ' concentration [micrograms/m3]')
        sub1.set_title('With background pollution')
        plt.legend(handles=handles)

        # plot differences

        station_y -= bgrd_y
        y -= bgrd_y
        ml_y -= bgrd_y

        min_y = int(np.min([np.min(y), np.min(ml_y)]))
        max_y = int(np.max([np.max(y), np.max(ml_y)]))

        # clip station data to range of Caline and ML output
        station_data = np.transpose([station_x, station_y])
        clipped_data = [tup for tup in station_data if min_y < tup[1] < max_y]
        [station_x, station_y] = np.transpose(clipped_data)

        sub2 = fig.add_subplot(2, 1, 2)
        station, = plt.plot(station_x, station_y, 'b.',
                            label='Station measurements')
        handles = [station]
        if poll in caline_time_series:
            caline, = plt.plot(x, y, 'r+', label='Caline estimates')
            handles.append(caline)
        if poll in ml_time_series:
            label = f'ML estimates ({kwargs["iteration"]}. iter)'
            mlp, = plt.plot(ml_x, ml_y, 'gx', label=label)
            handles.append(mlp)

        loc_x = list(np.linspace(x.min(), x.max(), 4))
        label_x = list(range(0, 4 * 24, 24))
        label_y = list(range(min(0, 2 * int((min_y - 1) / 2)),
                             max_y + 1,
                             2))
        plt.xticks(loc_x, label_x)
        plt.yticks(label_y, label_y)
        sub2.set_xticks(x, minor=True)
        sub2.grid(which='major')

        plt.xlabel('Hours from ' + date_start.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel(poll + ' concentration [micrograms/m3]')
        sub2.set_title('Without background pollution')
        plt.legend(handles=handles)

        # adjust spacing:
        st.set_y(0.97)
        fig.subplots_adjust(hspace=0.3)

        plot_name = (
            img_path + 'timeseries/timeseries_' + poll + '_'
            + str(site) + '_' + date_start.strftime('%Y-%m-%d_%H')
        )
        plt.savefig(plot_name + '.pdf')
        # plt.savefig(plot_name + '.png')
        plt.close()

    print('Done.')

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

    entry = collection_util.find({'contour_distance': 5})[0]
    entry = uda.db_util_entry_to_dict(entry)
    receptors_dict = entry['receptors_dict']

    for tile, recptors_list in receptors_dict.items():
        for receptor_coord in recptors_list:
            plot_timeseries(**get_parameters(receptor_coord))
