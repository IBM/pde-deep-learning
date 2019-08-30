from bson.son import SON
import datetime
import numpy as np
import pymongo

#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt


""" Script for plotting the weather data.

    This module plots aggregated weather data from the collection 
    'weather' of a MongoDB database called 'db_air_pollution' at port 
    27018. It writes plots into the relative img_path path.

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

Author: 
    Philipp Hähnel <phahnel@hsph.harvard.edu>

Last updated:
    2019 - 08 - 02

"""


###################
# Parameters to change
#
img_path = '../output/img/weather/'

plot_individual_weather_station_time_series = True
plot_speed = True
plot_dir = True
plot_dir_std = True
plot_temp = True

timestamp_start = datetime.datetime.strptime(
    '2017-07-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp()

figure_size = (20, 5)
#
#
###################


def get_weather_data(collection, site):
    """
        retrieve list of weather data from database
    :param collection: db_air_quality.collection_weather
    :param site: int
    :return: [time, speed, dir, std_dir, temp]
    """
    coll = collection.find({'site': site}
                           ).sort([('timestamp', pymongo.ASCENDING)])
    weather = [((e['timestamp'] - timestamp_start) / (60 * 60 * 24),
                e['wind_speed'] / 3.6,  # km/h to m/s
                e['wind_dir'],
                e['wind_dir_std'],
                e['temp']
                )
               for e in coll]
    return np.transpose(weather)


def plot_weather_data(station, data_x, data_y, ylabel, tag,
                      error_plot=False, err_data=None):
    """
        Plot weather data.
    :param station: int
    :param data_x: (list) time series
    :param data_y: (list)
    :param ylabel:
    :param tag: (str) for label and file name
    :param error_plot: (bool) True if error bar plot instead of normal plot
    :param err_data: (list) of error bars corresponding to data_y if error_plot==True
    :return:
    """
    fig = plt.figure(figsize=figure_size)
    sub = fig.add_subplot(1, 1, 1)
    if error_plot:
        line_up = plt.errorbar(data_x, data_y, yerr=err_data,
                               label=tag + ' at site ' + str(station),
                               fmt='r.')
    else:
        line_up, = plt.plot(data_x, data_y, 'r.',
                            label=tag + ' at site ' + str(station),
                            markersize=1)
    loc_x = data_x[0::24 * 7]
    label_x = [int(l) for l in data_x[0::24 * 7]]
    plt.xticks(loc_x, label_x)
    label_y = list(np.linspace(data_y.min(), data_y.max(), 10))
    plt.yticks(label_y, label_y)
    sub.grid(which='major')
    plt.xlabel('Days from 2017-07-01')
    plt.ylabel(ylabel)
    plt.legend(handles=[line_up])
    plt.savefig(img_path + tag + '_' + str(station) + '.pdf')
    plt.close()

    return None


def main():
    """
        Connect to database, retrieve weather data and plot for all
        available stations.
    :return: None
    """
    """ connect to internal Mongo database """
    client = pymongo.MongoClient('localhost', 27018)
    # 2017-07-01 01:00:00 to 2018-05-02 14:00:00
    collection_weather = client.db_air_quality.weather

    stations_pipeline = [{'$group': {'_id': '$site'}},
                         {'$sort': SON([('_id', 1)])}]
    stations = [site['_id']
                for site in collection_weather.aggregate(stations_pipeline)]

    for s, site in enumerate(stations):
        [time, speed, wdir, std_dir, temp] \
            = get_weather_data(collection_weather, site)
        if plot_individual_weather_station_time_series:
            if plot_speed:
                plot_weather_data(s, time, speed,
                                  ylabel='wind speed [m/s]',
                                  tag='wind speed')
            if plot_dir:
                plot_weather_data(s, time, wdir,
                                  ylabel='wind direction [degrees]; 0 = N',
                                  tag='wind direction',
                                  error_plot=True, err_data=std_dir)
            if plot_dir_std:
                plot_weather_data(s, time, std_dir,
                                  ylabel='wind direction std [degrees]',
                                  tag='wind direction std')
            if plot_temp:
                plot_weather_data(s, time, temp,
                                  ylabel='temperature [C]',
                                  tag='temperature')
    return None


if __name__ == '__main__':
    main()
