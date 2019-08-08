#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt

import pymongo

import util.util_db_access as uda
import util.util_measurements as um

""" Script to plot coordinate maps of traffic links, domain boundaries, 
    receptors, and measurement stations.  

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
        # list of ID's of tiles to be plotted:
        'sub_domain_selection': list(range(1, 12 + 1)),
        # 'sub_domain_selection': [6, 7],
        # if stations are plotted:
        'with_stations': True,
        # plot each of the runs in 'tags' individually or combined:
        'individual': False,
        # is appended to file name:
        'comment': ''
    }
    # runs selectors from the caline_estimates
    # distances = [
    #     5, 6, 7, 8, 9, 10, 11, 13, 17, 27, 37, 53, 71, 101, 103
    # ]  # Demo
    distances = [
        6, 11, 17, 20, 26, 33, 37, 47, 77, 105, 125, 160, 550, 600, 750
    ]  # Dublin
    param['tags'] = [{
            'run_tag': '2019-07-04',
            'contour_distance': dist,
            'case': 'Dublin'  # 'Demo' or 'Dublin'
        }
        for dist in distances
    ]
    # Title of each plot. If plots are combined, only the last title is used.
    param['titles'] = {dist: f'Map of Dublin.'
                       for dist in distances}
    return param


def plot_map(collection_util, stations, img_path='../output/img/', **kwargs):
    print(f'Plotting receptor map. Output in {img_path}')
    for tag in kwargs['tags']:
        util = uda.get_utilities_from_collection(collection_util, **tag)
        station_list = [point for point in stations.values()]
        borders = []
        for tile, ngbrs in util["intersections"].items():
            if tile in kwargs['sub_domain_selection']:
                for ngbr, border in ngbrs.items():
                    borders.append(border)
        print('Plotting map for current utilities.')
        handle_b = None
        handle_r = None
        handle_s = None
        handle_t = None
        # plot mesh grid
        for line in borders:
            y = [line[0][0], line[1][0]]
            x = [line[0][1], line[1][1]]
            handle_b, = plt.plot(x, y, c='k')
        for box_id in util["emitters_dict"]:
            if box_id in kwargs['sub_domain_selection']:
                data_e = util["emitters_dict"][box_id]
                data_r = util["receptors_dict"][box_id] \
                    if box_id in util["receptors_dict"] else []
                # plot line sources
                for line in data_e:
                    y = [line[0], line[2]]
                    x = [line[1], line[3]]
                    handle_t, = plt.plot(x, y, c='b')
                # plot receptors
                for point in data_r:
                    y = point[0]
                    x = point[1]
                    handle_r = plt.scatter(x, y, c='r', marker='.')
        # plot measurement stations
        for point in station_list:
            y = point[0]
            x = point[1]
            handle_s = plt.scatter(x, y, c='g', marker='D')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(kwargs['titles'][tag['contour_distance']])
        handles = [handle_b, handle_t, handle_r, handle_s] \
            if station_list else [handle_b, handle_t, handle_r]
        plt.legend(handles, ['domain border', 'street', 'receptor', 'station'])
        name = img_path + 'maps/discretization_' + str(tag) + kwargs['comment']
        if kwargs['individual']:
            plt.savefig(name + '.pdf')
            # plt.savefig(name + '.png')
            plt.close()
    if not kwargs['individual']:
        name = img_path + 'maps/discretization_' + kwargs['tags'][0]['case']
        if kwargs['comment']:
            name += '_' + kwargs['comment']
        plt.savefig(name + '.pdf')

    return None


if __name__ == '__main__':
    """ connect to internal Mongo database """
    client = pymongo.MongoClient('localhost', 27018)
    # 2017-07-01 01:00:00 to 2018-05-02 14:00:00
    collection_util = client.db_air_quality.util
    param = get_parameters()
    stations = um.get_stations() if param['with_stations'] else {}
    plot_map(collection_util=collection_util, stations=stations, **param)
