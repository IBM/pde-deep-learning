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
# import util.util_measurements as um


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
    2019 - 09 - 16

"""


def get_parameters():
    #  time slice (2017-07-01 01:00:00 to 2018-05-02 14:00:00)
    param = {
        'date_start': datetime.datetime(2017, 8, 7, 0),
        'period': datetime.timedelta(weeks=4),
        # If None, plots for receptors placed with 'contour_distance' are made.
        'receptor_coord': [53.34793989479807, -6.26312682482776],  # [53.34421064496467, -6.26476486860426],
        'station_name': 'Winetavern St Civic Offices',
        # Plot receptors that were placed at (multiples of) this distance.
        # For the Demo, the smallest distance is 5, for Dublin it is 6.
        'contour_distance': 5,
        'iteration': 10,
        'pollutants': ['PM10']
    }
    return param


def get_collections(port=27018):
    client = pymongo.MongoClient('localhost', port=port)
    collections = {'ml': client.db_air_quality.ml_estimates_cc_PM10,
                   'util': client.db_air_quality.util,
                   'station':client.db_air_quality.pollution_measurements,
                   'caline': client.db_air_quality.caline_estimates}
    return collections


def plot_timeseries(
        date_start, period, receptor_coord, station_name, pollutants,
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
    station_time_series = defaultdict(dict)
    polls = []
    for timestamp, pollution_values in station_measurements.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            if pollutant not in polls:
                polls.append(pollutant)
            station_time_series[pollutant][t] = value

    print('Getting background pollution data ...')
    background_pollution = uda.get_background_pollution(
        collection_measurements=collection_measurement,
        date_start=date_start,
        date_end=date_end
    )
    background_time_series = defaultdict(dict)
    for timestamp, pollution_values in background_pollution.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            background_time_series[pollutant][t] = value

    print('Getting Caline estimates ...')
    caline_estimates = uda.get_estimates_for_receptor(
        collection_estimates=collection_caline_estimates,
        date_start=date_start,
        date_end=date_end,
        receptor_coord=receptor_coord
    )
    caline_time_series = defaultdict(dict)
    timestamps = []
    for timestamp, pollution_values in caline_estimates.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            if t not in timestamps:
                timestamps.append(t)
            caline_time_series[pollutant][t] = value

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
    ml_time_series = defaultdict(dict)
    for timestamp, pollution_values in ml_estimates.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            ml_time_series[pollutant][t] = value

    timestamps.sort()

    print(f'Plotting time series for {site} ...')
    for p, poll in enumerate(pollutants):

        if poll not in caline_time_series or poll not in ml_time_series:
            continue

        station_x = np.array(list(station_time_series[poll].keys()))
        station_y = np.array(list(station_time_series[poll].values()))
        x = np.array(list(caline_time_series[poll].keys()))
        y = np.array(list(caline_time_series[poll].values()))
        ml_x = np.array(list(ml_time_series[poll].keys()))
        ml_y = np.array(list(ml_time_series[poll].values()))

        caline_time_series_plus = {
            time: caline_time_series[poll][time] + bg_y
            for time, bg_y in background_time_series[poll].items()
            if time in x
        }
        x_plus = np.array(list(caline_time_series_plus.keys()))
        y_plus = np.array(list(caline_time_series_plus.values()))
        ml_time_series_plus = {
            time: ml_time_series[poll][time] + bg_y
            for time, bg_y in background_time_series[poll].items()
            if time in ml_x
        }
        ml_x_plus = np.array(list(ml_time_series_plus.keys()))
        ml_y_plus = np.array(list(ml_time_series_plus.values()))

        min_y = np.min([np.min(y_plus), np.min(ml_y_plus), np.min(station_y)])
        max_y = np.max([np.max(y_plus), np.max(ml_y_plus), np.max(station_y)])

        # plot timeseries

        fig = plt.figure(figsize=(10, 10))
        # st = plt.suptitle('Pollution Data in Dublin City Center\n'
        #                   f'at receptor location {str(receptor_coord)},'
        #                   '\n'
        #                   'compared to measurement station '
        #                   f'at {um.get_stations()[station_name]}')

        sub1 = fig.add_subplot(2, 1, 1)
        station, = plt.plot(station_x, station_y, 'b.',
                            label='Station measurements')
        handles = [station]
        if poll in caline_time_series:
            caline, = plt.plot(x_plus, y_plus, 'r+', label='Caline estimates')
            handles.append(caline)
        if poll in ml_time_series:
            label = f'ML estimates'  # + f' ({kwargs["iteration"]}. iter)'
            mlp, = plt.plot(ml_x_plus, ml_y_plus, 'gx', label=label)
            handles.append(mlp)

        seconds_in_day = 60*60*24
        seconds_in_week = seconds_in_day * 7
        if period < datetime.timedelta(days=10):
            steps_x = seconds_in_day
            step = 1
        else:
            steps_x = seconds_in_week
            step = 7

        loc_x = list(np.arange(x_plus.min(), x_plus.max() + 2*60*60, steps_x))
        label_x = np.arange(date_start.strftime('%Y-%m-%d'), '2018-05-03',
                            dtype='datetime64[D]', step=step)
        label_y_min = 10 * int((min_y - 1) / 10)
        label_y_max = int(1.05 * max_y)
        steps = 10
        if label_y_max > 10:
            label_y = list(range(label_y_min, label_y_max, steps))
        else:
            label_y = np.linspace(label_y_min, label_y_max, 6)
        plt.xticks(loc_x, label_x)
        plt.yticks(label_y, label_y)
        sub1.set_xticks(x, minor=True)
        sub1.grid(which='major')

        # plt.xlabel('Hours from ' + date_start.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel(poll + ' concentration [$\mu g/m^3$]')
        sub1.set_title('With background pollution')
        plt.legend(handles=handles)

        # plot differences

        station_y = np.array([
            station_time_series[poll][time] - bg_y
            for time, bg_y in background_time_series[poll].items()
            if time in station_x
        ])

        min_y = np.min([np.min(y), np.min(ml_y)])
        max_y = np.max([np.max(y), np.max(ml_y)])

        sub2 = fig.add_subplot(2, 1, 2)
        handles = []

        # clip station data to range of Caline and ML output
        station_data = np.transpose([station_x, station_y])
        clipped_data = [tup for tup in station_data if min_y < tup[1] < max_y]
        if len(clipped_data):
            [station_x, station_y] = np.transpose(clipped_data)
            station, = plt.plot(station_x, station_y, 'b.',
                                label='Station measurements')
            handles.append(station)
        if poll in caline_time_series:
            caline, = plt.plot(x, y, 'r+', label='Caline estimates')
            handles.append(caline)
        if poll in ml_time_series:
            label = f'ML estimates'  # + f' ({kwargs["iteration"]}. iter)'
            mlp, = plt.plot(ml_x, ml_y, 'gx', label=label)
            handles.append(mlp)

        loc_x = list(np.arange(x.min(), x.max() + 2 * 60*60, steps_x))
        label_x = np.arange(date_start.strftime('%Y-%m-%d'), '2018-05-03',
                            dtype='datetime64[D]', step=step)
        label_y_min = 2 * int((min_y - 1) / 2)
        label_y_max = 1.05 * max_y
        steps = max(int((label_y_max - label_y_min) / 6), 1)
        if label_y_max > 10:
            label_y = list(range(label_y_min, int(label_y_max), steps))
        else:
            label_y = np.linspace(label_y_min, label_y_max, 6)
            label_y = [int(1000 * i) / 1000 for i in label_y]
        plt.xticks(loc_x, label_x)
        plt.yticks(label_y, label_y)
        sub2.set_xticks(x, minor=True)
        sub2.grid(which='major')

        # plt.xlabel('Hours from ' + date_start.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel(poll + ' concentration [$\mu g/m^3$]')
        sub2.set_title('Without background pollution')
        plt.legend(handles=handles)

        # adjust spacing:
        # st.set_y(0.97)
        fig.subplots_adjust(hspace=0.3)

        plot_name = (
            img_path + 'timeseries/timeseries_' + poll + '_'
            + date_start.strftime('%Y-%m-%d_%H') + '_' + str(site)
        )
        plt.savefig(plot_name + '.pdf')
        # plt.savefig(plot_name + '.png')
        plt.close()

    print('Done.')

    return None


if __name__ == '__main__':

    img_path = '../output/img/'

    """ connect to internal Mongo database """
    collections = get_collections()
    collection_measurement = collections['station']
    collection_caline_estimates = collections['caline']
    collection_estim = collections['ml']
    collection_util = collections['util']

    param = get_parameters()
    contour_distance = param['contour_distance']

    entry = collection_util.find({'contour_distance': contour_distance})[0]
    entry = uda.db_util_entry_to_dict(entry)
    receptors_dict = entry['receptors_dict']

    if param['receptor_coord'] is None:
        for tile, recptors_list in receptors_dict.items():
            for receptor_coord in recptors_list:
                param['receptor_coord'] = receptor_coord
                plot_timeseries(**param)
    else:
        plot_timeseries(**param)
