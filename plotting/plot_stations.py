from collections import defaultdict
from operator import itemgetter
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
    2019 - 09 - 16

"""


def get_parameters():
    #  time slice (2017-07-01 01:00:00 to 2018-05-02 14:00:00)
    param = {
        'date_start': datetime.datetime(2017, 7, 1, 0),
        # is appended to file name:
        'pollutants': ['NO2', 'PM10', 'PM25']
    }
    return param


def get_collections(port=27018):
    client = pymongo.MongoClient('localhost', port=port)
    collections = {
        'station':client.db_air_quality.pollution_measurements,
    }
    return collections


def plot_timeseries(
        date_start, station_name, pollutants,
        **kwargs
):

    timestamp_start = date_start.timestamp()
    date_end = datetime.datetime(2018, 5, 2, 0)

    days = date_end - date_start
    num_days = days.days

    print('Getting station measurement data ...')
    station_measurements = uda.get_station_measurements(
        collection_measurements=collection_measurement,
        date_start=date_start,
        date_end=date_end,
        station_name=station_name
    )
    station_time_series = defaultdict(list)
    polls = []
    timestamps = []
    for timestamp, pollution_values in station_measurements.items():
        for pollutant, value in pollution_values.items():
            t = timestamp - timestamp_start
            if pollutant not in polls:
                polls.append(pollutant)
            if t not in timestamps:
                timestamps.append(t)
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

    timestamps.sort()

    print(f'Plotting time series for {station_name} ...')
    for p, poll in enumerate(pollutants):

        if poll not in station_time_series:
            continue

        bgrd_time_series = sorted(background_time_series[poll], key=itemgetter(0))

        [bgrd_x, bgrd_y] = np.transpose(bgrd_time_series)
        [station_x, station_y] = np.transpose(station_time_series[poll])

        min_x = int(np.min([np.min(bgrd_x), np.min(station_x)]))
        max_x = int(np.max([np.max(bgrd_x), np.max(station_x)]))

        min_y = int(np.min([np.min(bgrd_y), np.min(station_y)]))
        max_y = int(np.max([np.max(bgrd_y), np.max(station_y)]))

        # plot timeseries

        fig = plt.figure(figsize=(200, 10))
        sub1 = fig.add_subplot(1, 1, 1)
        # st = plt.suptitle('Pollution Data in Dublin City Center\n'
        #                   f'at receptor location {str(receptor_coord)},'
        #                   '\n'
        #                   'compared to measurement station '
        #                   f'at {um.get_stations()[station_name]}')

        station, = plt.plot(station_x, station_y, 'b.',
                            label='Station measurements')
        bgrd, = plt.plot(bgrd_x, bgrd_y, 'g-',
                         label='Background values')
        handles = [station, bgrd]

        seconds_in_day = 60*60*24
        seconds_in_week = seconds_in_day * 7

        # 2017-07-01 was a Friday
        loc_x = list(
            np.arange(min_x + 2 * seconds_in_day, max_x, seconds_in_week)
        )
        label_x = np.arange('2017-07-03', '2018-05',
                            dtype='datetime64[D]', step=7)
        label_y = list(range(10 * int((min_y - 1) / 10), max_y + 1, 10))
        plt.xticks(loc_x, label_x)
        plt.yticks(label_y, label_y)
        loc_x_minor = list(np.arange(min_x, max_x, seconds_in_day))
        sub1.set_xticks(loc_x_minor, minor=True)
        sub1.grid(which='minor')
        sub1.grid(which='major', axis='x', linewidth=3)

        # plt.xlabel('Weeks from ' + date_start.strftime('%Y-%m-%d %H:%M:%S'))
        plt.ylabel(poll + ' concentration [$\mu g/m^3$]')
        plt.legend(handles=handles)

        fig.subplots_adjust(left=0.01, right=0.99)

        plot_name = (
            img_path + 'timeseries/Stations/timeseries_' + poll + '_'
            + station_name
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

    stations = um.get_stations()

    for station, coord in stations.items():
        plot_timeseries(station_name=station, **get_parameters())
