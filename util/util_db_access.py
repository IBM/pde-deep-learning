from collections import defaultdict

""" Utility methods for retrieving data from a MongoDB.

Description:
    This module implements utility functions for retrieving data from a 
    Mongo database.

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
    2019 - 08 - 06

"""


def get_weather_data(collection_weather, date_start, date_end):
    """
        Collects weather data from the collection_weather between the
        dates date_start and date_end.

    :param collection_weather: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :return: {timestamp: [wind_dir, wind_speed, wind_dir_std, temp]}
    """
    pipeline = [
        {'$match': {
            '$and': [
                {'timestamp': {'$gte': date_start.timestamp()}},
                {'timestamp': {'$lte': date_end.timestamp()}}
            ]
        }},
        {'$group': {
            '_id': {'timestamp': '$timestamp'},
            'wind_dir': {'$avg': '$wind_dir'},
            'wind_speed': {'$avg': '$wind_speed'},
            'wind_dir_std': {'$avg': '$wind_dir_std'},
            'temp': {'$avg': '$temp'}
        }}
    ]
    weather_data = {}
    for entry in collection_weather.aggregate(pipeline):
        weather_data[entry['_id']['timestamp']] = [entry['wind_dir'],
                                                   entry['wind_speed'] / 3.6,
                                                   # km/h to m/s
                                                   entry['wind_dir_std'],
                                                   entry['temp']]
    return weather_data


def get_traffic_volumes(collection_traffic, date_start, date_end, links):
    """
        Collects traffic data from the collection_traffic between the
        dates date_start and date_end. It aggregates the traffic volumes
        for all links.

    :param collection_traffic: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param links: list of tuples (start - end) of tuples (lat - lon)
    :return: {timestamp: {link: traffic_volume}}
    """
    traffic_pipeline = [
        {'$match': {
            '$and': [
                {'timestamp': {'$gte': date_start.timestamp()}},
                {'timestamp': {'$lte': date_end.timestamp()}}
            ]
        }},
        {'$project': {
            'timestamp': True,
            'vehicle_count': True,
            'link': {'$cond': [  # poor man's sort
                {'$lt': ['$site', '$next_site']},
                ['$site', '$next_site'],
                ['$next_site', '$site']
            ]}
        }},
        {'$match': {
            'link': {'$in': links}
        }},
        {'$group': {
            '_id': {'timestamp': '$timestamp',
                    'link': '$link'},
            'traffic_volume': {'$sum': '$vehicle_count'}
        }}]
    volumes = defaultdict(dict)
    for entry in collection_traffic.aggregate(traffic_pipeline,
                                              allowDiskUse=True):
        timestamp = entry['_id']['timestamp']
        link = tuple(entry['_id']['link'])
        volume = entry['traffic_volume']
        volumes[timestamp][link] = volume
    return volumes


def get_background_pollution(
        collection_measurements, date_start, date_end, method='avg'
):
    """
        Collects measurement data from the collection_measurement
        between the dates date_start and date_end. It aggregates the
        average value of all stations.

    :param collection_measurements: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param method:
    :return: {timestamp: {pollutant: value}}
    """
    pipeline = [
        {'$match': {
            '$and': [
                {'timestamp': {'$gte': date_start.timestamp()}},
                {'timestamp': {'$lte': date_end.timestamp()}},
                {'value': {'$gte': 0}}
            ]
        }},
        {'$group': {
            '_id': {'timestamp': '$timestamp',
                    'pollutant': '$pollutant'},
            'value': {'$' + method: '$value'}
        }}
    ]
    background = defaultdict(dict)
    for entry in collection_measurements.aggregate(pipeline):
        timestamp = entry['_id']['timestamp']
        pollutant = entry['_id']['pollutant']
        value = entry['value']
        background[timestamp][pollutant] = value
    return background


def get_station_measurements(
        collection_measurements, date_start, date_end, station_name
):
    """
        Collects measurement data from the collection_measurement
        between the dates date_start and date_end for a specific station.

    :param collection_measurements: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param station_name: name of the station
    :return: {timestamp: {pollutant: value}}
    """
    pipeline = [
        {'$match': {
            '$and': [
                {'timestamp': {'$gte': date_start.timestamp()}},
                {'timestamp': {'$lte': date_end.timestamp()}},
                {'value': {'$gte': 0}},
                {'site': station_name}
            ]
        }}
    ]
    measurements = defaultdict(dict)
    for entry in collection_measurements.aggregate(pipeline):
        timestamp = entry['timestamp']
        pollutant = entry['pollutant']
        value = entry['value']
        measurements[timestamp][pollutant] = value
    return measurements


def get_estimates(
        collection_estimates, date_start, date_end, **kwargs
):
    """
        Collects pollution estimate data from the
        collection_estimates between the dates date_start and
        date_end of a the run corresponding to filters set in kwargs

    :param collection_estimates: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param kwargs: Caline run identifiers
    :return: caline_estimates = {timestamp: {coord: {pollutant: value}}}
             receptor_list = [coord]
    """
    estimates = {}  # see docstring
    receptor_list = []  # [(lat, lon)]

    filter_list = [
        {'timestamp': {'$gte': date_start.timestamp()}},
        {'timestamp': {'$lte': date_end.timestamp()}}
    ]
    filter_list += [{k: v} for k, v in kwargs.items()]

    pipeline = [{'$match': {'$and': filter_list}}]

    for entry in collection_estimates.aggregate(pipeline):
        timestamp = entry['timestamp']
        coord = tuple(entry['coord'])
        poll = entry['pollutant']
        if timestamp not in estimates:
            estimates[timestamp] = {}
        if coord not in estimates[timestamp]:
            estimates[timestamp][coord] = {}
        estimates[timestamp][coord][poll] = entry['value']

        if coord not in receptor_list:
            receptor_list.append(coord)

    return estimates, receptor_list


def get_estimates_for_receptor(
        collection_estimates, date_start, date_end,
        receptor_coord, **kwargs
):
    """
        Collects pollution estimate data from the
        collection_estimates between the dates date_start and
        date_end of a the run corresponding to filters set in kwargs

    :param collection_estimates: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param kwargs: Caline run identifiers
    :param receptor_coord: [lat, lon]
    :return: caline_estimates = {timestamp: {pollutant: value}}
    """
    estimates = defaultdict(dict)  # see docstring

    filter_list = [
        {'timestamp': {'$gte': date_start.timestamp()}},
        {'timestamp': {'$lte': date_end.timestamp()}},
        {'coord': receptor_coord}
    ]
    filter_list += [{k: v} for k, v in kwargs.items()]

    pipeline = [{'$match': {'$and': filter_list}}]

    for entry in collection_estimates.aggregate(pipeline):
        timestamp = entry['timestamp']
        poll = entry['pollutant']
        estimates[timestamp][poll] = entry['value']

    return estimates


def db_util_entry_to_dict(entry, tiles=None):
    """
        Reshapes the retrieved db utilities collection entry into the
        format used by the scripts.

    :param entry: utilities dict that was retrieved from a mongo db collection
    :param tiles: list of tiles
    :return: utilities dict as used by all scripts.
    """
    util = dict()
    if tiles is None:
        tiles = [tile[0] for tile in entry['domain_dict']]

    # different treatment for different entries
    tuple_list = ['links_in_area', 'links_dict',
                  'receptors_index']
    dict_list = ['emitters_dict', 'emitters_dict_cart',
                 'receptors_dict', 'receptors_dict_cart',
                 'norm_emitters', 'norm_receptors',
                 'domain_neighbors']
    dict_dict_list = ['intersections']

    for key, value in entry.items():
        if key == '_id':
            continue
        elif key in tuple_list:
            util[key] = {tuple(k): v for k, v in value}
        elif key in dict_list:
            util[key] = {k: v for k, v in value if k in tiles}
        elif key in dict_dict_list:
            util[key] = {k: {kv: vv for kv, vv in v if kv in tiles}
                         for k, v in value if k in tiles}
        elif key == 'domain_dict':
            util[key] = {
                k: {kv: ([tuple(l) for l in vv] if kv == 'links' else vv)
                    for kv, vv in v.items()}
                for k, v in value if k in tiles
            }
        else:  # if key == 'run_tag' or key == 'bounding_box' or something else
            util[key] = value

    # update link entries to select only the ones that are in the
    # selected tiles
    util['links_in_area'] = {
        link: coords
        for link, coords in util['links_in_area'].items()
        if any([link in util['domain_dict'][tile]['links'] for tile in tiles])
    }
    util['links_dict'] = {}
    for tile, values in util['domain_dict'].items():
        for link in values['links']:
            util['links_dict'][tuple(link)] = [tile]

    return util


def util_dict_to_db_entry(util):
    """
        Reshapes the utility dict into a dict over lists as writing the
        util dict into a mongo db throws the error
        bson.errors.InvalidDocument: documents must have only string keys!

    :param util: utility dictionary
    :return: dictionary that can be inserted into a mongo db.
    """
    dict_list = ['links_in_area', 'links_dict',
                 'emitters_dict', 'emitters_dict_cart',
                 'receptors_dict', 'receptors_dict_cart',
                 'norm_emitters', 'norm_receptors',
                 'domain_neighbors', 'intersections',
                 'domain_dict', 'receptors_index']
    entry = {
        key: ([
            (k, [(kk, vv) for kk, vv in v.items()]
                if key == 'intersections' else v
             )
            for k, v in values.items()
            ]
            if key in dict_list else values
        )
        for key, values in util.items()}
    return entry


def get_utilities_from_collection(collection_util, **kwargs):
    """
        Retrieves utility information from collection.
        Assumes that kwarg tags give unique identifiers for the utility
        information. If multiple entries are found, the first one is
        returned.

    :param collection_util: MongoDB collection
    :return: (dict) of utility information
    """
    print(f'Searching for utilities for the run with tags {kwargs}.')
    entry = collection_util.find(kwargs)
    if entry.count():
        utilities = db_util_entry_to_dict(entry[0])
        print('Utilities retrieved from database.')
    else:
        print('No utilities entry found in database.')
        utilities = {}
    return utilities


def get_pre_processed_data(collection_pre_proc, domains):
    data = {}
    for sub_domain_id in domains:
        data[sub_domain_id] = defaultdict(list)
        for entry in collection_pre_proc.find({'domain': sub_domain_id}):
            data[sub_domain_id]['input'].append(entry['input'])
            data[sub_domain_id]['labels'].append(entry['labels'])
    return data
