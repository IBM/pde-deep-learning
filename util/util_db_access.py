from collections import defaultdict

""" Utility methods for retrieving data from a MongoDB.

Description:
    This module implements utility functions for retrieving data from a Mongo database.

-*- coding: utf-8 -*-

Legal:
    (C) Copyright IBM 2018.
    
    This code is licensed under the Apache License, Version 2.0. You may
    obtain a copy of this license in the LICENSE.txt file in the root directory
    of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
    
    Any modifications or derivative works of this code must retain this
    copyright notice, and modified files need to carry a notice indicating
    that they have been altered from the originals.

    IBM-Review-Requirement: Art30.3
    Please note that the following code was developed for the project VaVeL at
    IBM Research -- Ireland, funded by the European Union under the
    Horizon 2020 Program.
    The project started on December 1st, 2015 and was completed by December 1st,
    2018. Thus, in accordance with Article 30.3 of the Multi-Beneficiary General
    Model Grant Agreement of the Program, there are certain limitations in force 
    up to December 1st, 2022. For further details please contact Jakub Marecek
    (jakub.marecek@ie.ibm.com) or Gal Weiss (wgal@ie.ibm.com).

If you use the code, please cite our paper:
https://arxiv.org/abs/1810.09425

Authors:
    Philipp Hähnel <phahnel@hsph.harvard.edu>

Last updated:
    2019 - 05 - 05

"""


def get_weather_data(collection_weather, date_start, date_end):
    """
        Collects weather data from the collection_weather between the dates
        date_start and date_end.
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
                                                   entry['wind_speed'],
                                                   entry['wind_dir_std'],
                                                   entry['temp']]
    return weather_data


def get_traffic_volumes(collection_traffic, date_start, date_end, links):
    """
        Collects traffic data from the collection_traffic between the dates
        date_start and date_end. It aggregates the traffic volumes for all links.
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
    for entry in collection_traffic.aggregate(traffic_pipeline, allowDiskUse=True):
        timestamp = entry['_id']['timestamp']
        link = tuple(entry['_id']['link'])
        volume = entry['traffic_volume']
        volumes[timestamp][link] = volume
    return volumes


def get_background_pollution(collection_measurements, date_start, date_end):
    """
        Collects measurement data from the collection_measurement between the dates
        date_start and date_end. It aggregates the average value of all stations.
    :param collection_measurements: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :return: {timestamp: {pollutant: average value}}
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
            'value': {'$avg': '$value'}
        }}
    ]
    background = defaultdict(dict)
    for entry in collection_measurements.aggregate(pipeline):
        timestamp = entry['_id']['timestamp']
        pollutant = entry['_id']['pollutant']
        value = entry['value']
        background[timestamp][pollutant] = value
    return background


def get_station_measurements(collection_measurements, date_start, date_end, station_name):
    """
        Collects measurement data from the collection_measurement between the dates
        date_start and date_end for a specific station
    :param collection_measurements: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param station_name: name of the station
    :return: {timestamp: {pollutant: average value}}
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


def get_caline_estimates(collection_caline_estimates, date_start, date_end, run_tag,
                         receptor_coords=None):
    """
        Collects pollution estimate data from the collection_pollution_estimates between
        the dates date_start and date_end of a the run corresponding to run_tag
    :param collection_caline_estimates: MongoDB collection
    :param date_start: datetime object
    :param date_end: datetime object
    :param run_tag: identifier for Caline run
    :param receptor_coords: (optional) if not None, it returns the timeline of Caline estimates
                            for that receptor coordinate only.
    :return: if receptor_coords is None:
                 caline_estimates = {timestamp: {coord: {pollutant: value}}}
             else:
                 caline_estimates = {timestamp: {pollutant: value}}
             receptor_list = [(lat, lon)]
    """
    caline_estimates = {}  # see docstring
    receptor_list = []  # [(lat, lon)]

    if receptor_coords is None:
        pipeline = [
            {'$match': {
                '$and': [
                    {'timestamp': {'$gte': date_start.timestamp()}},
                    {'timestamp': {'$lte': date_end.timestamp()}},
                    {'run_tag': run_tag}
                ]
            }}
        ]
    else:
        pipeline = [
            {'$match': {
                '$and': [
                    {'timestamp': {'$gte': date_start.timestamp()}},
                    {'timestamp': {'$lte': date_end.timestamp()}},
                    {'run_tag': run_tag},
                    {'coord': receptor_coords}
                ]
            }}
        ]

    for caline_entry in collection_caline_estimates.aggregate(pipeline):
        timestamp = caline_entry['timestamp']
        coord = tuple(caline_entry['coord'])
        poll = caline_entry['pollutant']
        if timestamp not in caline_estimates:
            caline_estimates[timestamp] = {}
        if receptor_coords is None:
            if coord not in caline_estimates[timestamp]:
                caline_estimates[timestamp][coord] = {}
            caline_estimates[timestamp][coord][poll] = caline_entry['value']
        else:
            caline_estimates[timestamp][poll] = caline_entry['value']

        if coord not in receptor_list:
            receptor_list.append(coord)

    return caline_estimates, receptor_list


def get_utilities_from_collection(collection_util, run_tag=None, mesh_size=12):
    """
        Retrieves utility information from collection.
    :param collection_util: MongoDB collection
    :param run_tag: (str)
    :param mesh_size: (int) number of sub-domains, i.e. length of domain_dict
    :return: (dict) of utility information
    """
    if run_tag is not None:
        util = collection_util.find({'run_tag': run_tag})
    else:
        util = collection_util.find({'domain_dict': {'$size': mesh_size}})
    utilities = dict()
    if util.count():
        for key, value in util[0].items():
            if key == '_id':
                continue
            elif key == 'run_tag' or key == 'bounding_box':
                utilities[key] = value
            elif key == 'intersections':
                utilities[key] = {tuple(k) if isinstance(k, list) else k: {kv: vv for kv, vv in v}
                                  for k, v in value}
            else:
                utilities[key] = {tuple(k) if isinstance(k, list) else k: v
                                  for k, v in value}

        utilities['domain_dict'] = {id: {key: ([tuple(link) for link in lst] if key == 'links' else lst)
                                         for key, lst in tile.items()}
                                    for id, tile in utilities['domain_dict'].items()}
    else:
        print('No utilities entry found in database.')
    return utilities


def get_pre_processed_data(collection_pre_proc, domains):
    data = {}
    for sub_domain_id in domains:
        data[sub_domain_id] = defaultdict(list)
        for entry in collection_pre_proc.find({'domain': sub_domain_id}):
            data[sub_domain_id]['input'].append(entry['input'])
            data[sub_domain_id]['labels'].append(entry['labels'])
    return data
