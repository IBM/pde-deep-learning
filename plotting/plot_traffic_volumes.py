import datetime
import numpy as np
import pymongo
import sys

#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib

    matplotlib.use('PS')
    from matplotlib import pyplot as plt

from util import util_db_access as uda


""" Script for plotting the traffic data.

    This module plots aggregated traffic data from the collection 
    'traffic_volumes' of a MongoDB database called 'db_air_pollution' 
    at port 27018. It writes plots into the relative img_path path.

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


###################
# Parameters to change
#
img_path = '../output/img/traffic/'

# 2017-07-01 01:00:00 to 2018-05-02 14:00:00
date_start = datetime.datetime(2017, 7, 1, 0)
date_end = datetime.datetime(2018, 5, 2, 23)
timestamp_start = datetime.datetime.strptime(
    '2017-07-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp()

figure_size = (35, 5)
#
#
###################


def get_traffic_data(collection, domain_dict):
    """
        Retrieve list of link sources as a dictionary from database.
        It's a rearranged output of util.util_caline.get_traffic_volumes()
    :param collection: db_air_quality.traffic_volumes
    :param domain_dict: from db_air_quality.util
    :return: link_time_series = {link: {timestamp: volume}}
    """
    traffic_links = [
        sorted(link)
        for box in domain_dict.values()
        for link in box['links']
    ]
    traffic_volumes = uda.get_traffic_volumes(
        collection, date_start, date_end, traffic_links
    )
    link_time_series = {}
    for timestamp, link_volumes in traffic_volumes.items():
        for link, volume in link_volumes.items():
            if link not in link_time_series:
                link_time_series[link] = {}
            link_time_series[link][timestamp] = volume
    return link_time_series


def plot_traffic(time_series, link):
    """
        Plots the time_series data
    :param time_series: {timestamp: volume}
    :param link: int
    :return: None
    """
    x = np.array([float(i) - timestamp_start for i in time_series.keys()])
    y = np.array([float(i) for i in time_series.values()])

    fig = plt.figure(figsize=figure_size)
    sub = fig.add_subplot(1, 1, 1)
    line_up, = plt.plot(x, y, label='traffic data at link ' + str(link))
    # num = max(1, int((x.max() - x.min()) / (60 * 60 * 24 * 7)))
    loc_x = np.linspace(x.min(), x.max(),
                        max(1, int((x.max() - x.min())
                                   / (60 * 60 * 24 * 7))
                            )
                        )
    # loc_x = list(range(int(x.min()), int(x.max()),
    #                    max(1, int((x.max() - x.min())
    #                               / (60 * 60 * 24 * 7))
    #                        )
    #                    ))
    label_x = list(range(int((x.max() - x.min()) / (60 * 60 * 24 * 7))))
    plt.xticks(loc_x, label_x)
    # sub.set_xticks(x, minor=True)
    label_y = list(np.linspace(y.min(), y.max(), 10))
    # label_y = list(range(int(y.min()), int(y.max()),
    #                      max(1, int((y.max() - y.min()) / 5))))
    if len(label_y) < 5:
        label_y = list(set(y))
    plt.yticks(label_y, label_y)
    sub.grid(which='major')
    plt.xlabel('Weeks from 2017-07-01')
    plt.ylabel('vehicle count')
    plt.legend(handles=[line_up])
    plt.savefig(img_path + 'traffic_volumes_' + str(link) + '.pdf')
    plt.close()

    return None


def main():
    """
        Connect to database, retrieve traffic data and plot for all
        available roads.

    :return: None
    """
    """ connect to internal Mongo database """
    client = pymongo.MongoClient('localhost', 27018)
    # 2017-07-01 01:00:00 to 2018-05-02 14:00:00
    collection_traffic_volumes = client.db_air_quality.traffic_volumes
    collection_util = client.db_air_quality.util

    util = collection_util.find({})

    if util.count():
        entry = util[0]
        domain_dict = {k: v for k, v in entry['domain_dict']}
    else:
        print('Update utility DB!')
        sys.exit()

    print('getting traffic data ...')
    link_time_series = get_traffic_data(collection_traffic_volumes,
                                        domain_dict)

    print('plotting traffic data ...')
    for link, time_series in link_time_series.items():
        plot_traffic(time_series, link)

    return None


if __name__ == '__main__':
    main()
