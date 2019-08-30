from collections import defaultdict
import datetime

from matplotlib.colors import Normalize
#  This runs into an issue under Mac OS X, thus the workaround.
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('PS')
    from matplotlib import pyplot as plt

import numpy as np
import pymongo
import scipy as sp
from scipy.interpolate import griddata

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
    2019 - 08 - 30

"""


def get_parameters():
    """
    :return:
    """
    #  time slice (2017-07-01 01:00:00 to 2018-05-02 14:00:00)
    param = {
        'case': 'Demo',
        'date_start': datetime.datetime(2017, 10, 22, 0),
        'date_end': datetime.datetime(2017, 10, 23, 0),
        'iteration': 1,
        'pollutants': ['NO2'],
        'resolution': 150,
    }
    return param


class MidpointNormalize(Normalize):
    """ from https://matplotlib.org/users/colormapnorms.html """
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(
            0,
            1 / 2 * (1 - abs((self.midpoint - self.vmin)
                             / (self.midpoint - self.vmax)))
        )
        normalized_max = min(
            1,
            1 / 2 * (1 + abs((self.vmax - self.midpoint)
                             / (self.midpoint - self.vmin)))
        )
        normalized_mid = 0.5
        x = [self.vmin, self.midpoint, self.vmax]
        y = [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(sp.interp(value, x, y))


def plot_heatmap(date, **kwargs):
    print('Getting Caline estimates ...')
    caline_estimates, receptor_list = uda.get_estimates(
        collection_estimates=collection_caline_estimates,
        date_start=date,
        date_end=date,
        case=kwargs['case']
    )
    caline_data = defaultdict(list)
    caline_dict = defaultdict(dict)  # just for convenience
    # turn {timestamp: {coord: {pollutant: value}}}
    # into {pollutant: [coord.x, coord.y, value]}
    for _, receptor_values in caline_estimates.items():
        for coord, pollution_values in receptor_values.items():
            for pollutant, value in pollution_values.items():
                caline_data[pollutant].append(list(coord) + [value])
                caline_dict[pollutant][coord] = value

    print('Getting MLP estimates ...')
    ml_filter = {
        'iteration': kwargs['iteration']
        # 'settings.gamma': gamma,
        # 'settings.kappa': kappa,
        # 'settings': iter
    }
    ml_estimates, receptor_list = uda.get_estimates(
        collection_estimates=collection_estim,
        date_start=date,
        date_end=date,
        **ml_filter
    )
    ml_data = defaultdict(list)
    diff = defaultdict(list)
    # turn {timestamp: {coord: {pollutant: value}}}
    # into {pollutant: [coord.x, coord.y, value]}
    for _, receptor_values in ml_estimates.items():
        for coord, pollution_values in receptor_values.items():
            for pollutant, value in pollution_values.items():
                ml_data[pollutant].append(list(coord) + [value])
                diff[pollutant].append(
                    list(coord) + [caline_dict[pollutant][coord] - value]
                )

    print('Plotting heatmap ...')
    for p, poll in enumerate(kwargs['pollutants']):

        xi, yi, zi = [], [], []
        ml_xi, ml_yi, ml_zi = [], [], []
        d_xi, d_yi, d_zi = [], [], []

        if poll in caline_data:
            [y, x, z] = np.transpose(caline_data[poll])
            xi = np.linspace(min(x), max(x), kwargs['resolution'])
            yi = np.linspace(min(y), max(y), kwargs['resolution'])
            zi = griddata((x, y), z,
                          (xi[None, :], yi[:, None]),
                          method='nearest')
        if poll in ml_data:
            [ml_y, ml_x, ml_z] = np.transpose(ml_data[poll])
            ml_xi = np.linspace(min(ml_x), max(ml_x), kwargs['resolution'])
            ml_yi = np.linspace(min(ml_y), max(ml_y), kwargs['resolution'])
            ml_zi = griddata((ml_x, ml_y), ml_z,
                             (ml_xi[None, :], ml_yi[:, None]),
                             method='nearest')
        if poll in diff:
            [d_y, d_x, d_z] = np.transpose(diff[poll])
            d_xi = np.linspace(min(d_x), max(d_x), kwargs['resolution'])
            d_yi = np.linspace(min(d_y), max(d_y), kwargs['resolution'])
            d_zi = griddata((d_x, d_y), d_z,
                            (d_xi[None, :], d_yi[:, None]),
                            method='nearest')

        # Layout of the plot:
        #
        # Caline estimates. differences
        # position map    . ML estimates

        fig = plt.figure(figsize=(15, 10))
        st = plt.suptitle(
            'Pollution estimates for ' + kwargs['case'] + ' for ' + str(date)
            + '\n'
            + poll + ' concentration levels in [micrograms/m3]',
            fontsize='x-large'
        )

        sub1 = fig.add_subplot(2, 2, 1)
        sub1.set_title('Caline estimates')
        plt.contour(xi, yi, zi, 10, linewidths=0.5, colors='k')
        plt.contourf(xi, yi, zi, 15, cmap='YlOrRd')
        plt.colorbar()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        sub2 = fig.add_subplot(2, 2, 2)
        sub2.set_title('Differences between Caline and ML estimates')
        plt.contour(d_xi, d_yi, d_zi, 10, linewidths=0.5, colors='k')
        plt.contourf(d_xi, d_yi, d_zi, 15,
                     cmap='seismic',
                     norm=MidpointNormalize(
                         vmin=np.min(d_zi),
                         vmax=np.max(d_zi),
                         midpoint=0
                     ))
        plt.colorbar()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # sub3 = fig.add_subplot(2, 2, 3)
        # plot_receptor_map()

        sub4 = fig.add_subplot(2, 2, 4)
        sub4.set_title('ML estimates after ' + str(kwargs['iteration'])
                       + ' iteration(s)')
        plt.contour(ml_xi, ml_yi, ml_zi, 10, linewidths=0.5, colors='k')
        plt.contourf(ml_xi, ml_yi, ml_zi, 15, cmap='YlOrRd')
        plt.colorbar()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85, hspace=0.3)

        plt.savefig(
            img_path + 'contour_maps/' + poll + '/'
            + kwargs['case']
            + '_' + str(kwargs['iteration'])
            + '_' + date.strftime('%Y-%m-%d_%H')
            + '.pdf'
        )
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

    parameters = get_parameters()

    date_start = parameters['date_start']
    date_end = parameters['date_end']
    time_step = datetime.timedelta(hours=1)

    current = date_start

    while current <= date_end:
        plot_heatmap(
            date=current,
            **parameters
        )
        current += time_step
