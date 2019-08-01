from collections import defaultdict
import geopy.distance
import numpy as np
import operator
import sys
import utm

import util.util_db_access as uda
from util.util_measurements import get_stations
from run.create_demo import create_demo

""" Utility methods for decomposing the domain in sub-domains.

Description:
    This module implements utility functions to decompose the main 
    domain into smaller sub-domains, as well as methods to place 
    receptors and get the emitters. The method get_utilities() at the 
    end retrieves all utility information for the specified parameters. 

    This module fetches utility information for a run from the database. 
    If this information is not available, it will be generated and 
    stored as a new entry in a collection 'util' of a MongoDB database 
    called  'db_air_quality' at port 27018.

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
    Julien Monteil <Julien.Monteil@ie.ibm.com>

Last updated:
    2019 - 08 - 01

"""


def select_links_in_area(links_dict, bounding_box):
    coord_min = bounding_box['coord_min']
    coord_max = bounding_box['coord_max']
    links_in_area = {}
    for key, link in links_dict.items():
        lat1 = link[0][0]
        lon1 = link[0][1]
        lat2 = link[1][0]
        lon2 = link[1][1]
        if ((coord_min[0] <= lat1 <= coord_max[0]
             and coord_min[1] <= lon1 <= coord_max[1])
                or (coord_min[0] <= lat2 <= coord_max[0]
                    and coord_min[1] <= lon2 <= coord_max[1])):
            links_in_area[key] = link
    return links_in_area


def is_link_in_area(link, coord_min, coord_max):
    lat1 = link[0][0]
    lon1 = link[0][1]
    lat2 = link[1][0]
    lon2 = link[1][1]
    if ((coord_min[0] <= lat1 <= coord_max[0]
         and coord_min[1] <= lon1 <= coord_max[1])
            or (coord_min[0] <= lat2 <= coord_max[0]
                and coord_min[1] <= lon2 <= coord_max[1])):
        return True
    else:
        return False


def is_point_in_area(point, box):
    if (box[0][0] <= point[0] <= box[1][0]
            and box[0][1] <= point[1] <= box[1][1]):
        return True
    else:
        return False


def decompose_domain(coord_min, coord_max, links_in_area):
    """
        Decomposes the domain into rectangular areas with a
        maximum of 20 links.
        On the Dublin data, this results in this mesh:

        |-----------------------|
        |       |       |  12   |
        |   4   |   8   |-------|
        |       |       |  11   |
        |-------|       |-------|
        |   3   |-------|       |
        |-------|   7   |       | longitude
        |       |       |  10   |
        |   2   |-------|       |
        |-------|   6   |       |
        |       |-------|       |
        |   1   |   5   |-------|
        |       |       |   9   |
        |-----------------------|
                latitude

        with {sub-domain: number of links} as
        {1: 19, 2: 15, 3: 16, 4: 11, 5: 17, 6: 20, 7: 16, 8: 12, 9: 16,
        10: 15, 11: 15, 12: 7}.
        The Demo example uses tiles 6 and 7 with re-aligned and
        additional links.

    :param coord_min:
    :param coord_max:
    :param links_in_area: {(site, next_site): [[lat, lon], [lat, lon]}
    :return: links_dict: {(site, next_site): [sub_domain_id], ...}
             domain_dict: {sub_domain_id: {'coord': [[lat, lon], [lat, lon]],
                                 'links': [(site, next_site), ... ]}}
    """

    nb_links_per_sub_area = 52  # math.ceil(len(links_in_area)/3)
    # Here i could not think of a better way,
    # I played around to get a 3x4 grid...
    index = 100

    """ intermediate dictionaries """
    links_sub_dict = {key: [] for key in links_in_area.keys()}

    lon = np.linspace(coord_min[1], coord_max[1], index)
    counter = 0
    box_sub_dict = {1: {'coord': [],
                        'links': []}}
    box_sub_id = 1
    lon_lat_min = coord_min
    coord_max_temp = coord_max
    for i in range(index - 1):
        coord_min_temp = np.asarray([coord_min[0], lon[i]])
        coord_max_temp = np.asarray([coord_max[0], lon[i + 1]])
        for link, link_coords in links_in_area.items():
            link_in_area = is_link_in_area(link_coords, coord_min_temp,
                                           coord_max_temp)
            if link_in_area and box_sub_id not in links_sub_dict[link]:
                counter += 1
                box_sub_dict[box_sub_id]['links'].append(link)
                links_sub_dict[link].append(box_sub_id)

        if counter >= nb_links_per_sub_area:
            box_sub_dict[box_sub_id]['coord'] = [lon_lat_min, coord_max_temp]
            lon_lat_min = np.asarray([coord_min[0], lon[i + 1]])
            counter = 0
            box_sub_id += 1
            box_sub_dict[box_sub_id] = {}
            box_sub_dict[box_sub_id]['coord'] = []
            box_sub_dict[box_sub_id]['links'] = []
    if len(box_sub_dict[box_sub_id]['coord']) == 0:
        box_sub_dict[box_sub_id]['coord'] = [lon_lat_min, coord_max_temp]

    """final dictionaries"""
    links_dict = {}
    for key in links_in_area:
        links_dict[key] = []
    domain_dict = {}
    sub_domain_id = 1
    domain_dict[sub_domain_id] = {}
    domain_dict[sub_domain_id]['coord'] = []
    domain_dict[sub_domain_id]['links'] = []
    lat = np.linspace(coord_min[0], coord_max[0], index)
    counter = 0

    for key in box_sub_dict:
        sub_dict = box_sub_dict[key]
        lon_lat_min = sub_dict['coord'][0]
        nb_links_per_area = 15  # math.ceil(len(dict['links'])/3)
        # Here i could not think of a better way,
        # I played around to get a 3x4 grid...
        for i in range(0, index - 1):
            coord_min_temp = np.asarray([lat[i], sub_dict['coord'][0][1]])
            coord_max_temp = np.asarray([lat[i + 1], sub_dict['coord'][1][1]])
            for link in sub_dict['links']:
                link_in_area = is_link_in_area(links_in_area[link],
                                               coord_min_temp, coord_max_temp)
                if link_in_area and sub_domain_id not in links_dict[link]:
                    counter += 1
                    domain_dict[sub_domain_id]['links'].append(link)
                    links_dict[link].append(sub_domain_id)

            if nb_links_per_area <= counter <= 20:
                domain_dict[sub_domain_id]['coord'] = [list(lon_lat_min),
                                                       list(coord_max_temp)]
                lon_lat_min = np.asarray([lat[i + 1], sub_dict['coord'][0][1]])
                counter = 0
                sub_domain_id += 1
                domain_dict[sub_domain_id] = {}
                domain_dict[sub_domain_id]['coord'] = []
                domain_dict[sub_domain_id]['links'] = []
        if len(domain_dict[sub_domain_id]['coord']) == 0:
            domain_dict[sub_domain_id]['coord'] = [list(lon_lat_min),
                                                   list(coord_max_temp)]
            if len(domain_dict.keys()) < 12:
                counter = 0
                sub_domain_id += 1
                domain_dict[sub_domain_id] = {}
                domain_dict[sub_domain_id]['coord'] = []
                domain_dict[sub_domain_id]['links'] = []
    return links_dict, domain_dict


def get_neighboring_domains(domain_dict):
    """
    :param domain_dict:
    :return: neighbors = {tile: [neighbors]},
             intersections = {tile: {neighbor: line}}
    """

    def get_edges(min_max_list):
        # decorate with list to avoid changing domain_dict
        corners = list(min_max_list)
        corners.append([min_max_list[0][0], min_max_list[1][1]])
        corners.append([min_max_list[1][0], min_max_list[0][1]])
        corners = [corners[index] for index in [0, 2, 1, 3]]
        corners.append(corners[0])
        tile_edges = []
        for i in range(len(corners) - 1):
            tile_edges.append([corners[i], corners[i + 1]])
        return tile_edges

    neighbors = defaultdict(list)
    intersections = defaultdict(dict)
    for sub_domain_id, sub_domain in domain_dict.items():
        edges = get_edges(sub_domain['coord'])
        for other_sub_domain_id, other_sub_domain in domain_dict.items():
            if other_sub_domain_id != sub_domain_id:
                other_edges = get_edges(other_sub_domain['coord'])
                intersection = []
                for edge in edges:
                    for other_edge in other_edges:
                        for point in edge:
                            if (distance_point_from_line(point, other_edge) == 0
                                    and point not in intersection):
                                intersection.append(list(point))
                        for point in other_edge:
                            if (distance_point_from_line(point, edge) == 0
                                    and point not in intersection):
                                intersection.append(list(point))
                if len(intersection) > 1:
                    neighbors[sub_domain_id].append(other_sub_domain_id)
                    intersections[sub_domain_id][other_sub_domain_id] \
                        = intersection
    return neighbors, intersections


def distance_point_from_line(point, line):
    """
        returns the distance from a point to a line segment

    :param point: list of coordinates
    :param line: list for start and endpoint of coordinates
    :return: distance in meters
    """
    start = np.asarray(line[0])
    end = np.asarray(line[1])
    start_to_end = np.subtract(end, start)
    line_length = float(np.linalg.norm(start_to_end))
    line_unit_vec = start_to_end / line_length

    start_to_point = np.subtract(np.asarray(point), start)

    offset = np.dot(start_to_point, line_unit_vec) / line_length
    if offset > 1:
        offset = 1
    elif offset < 0:
        offset = 0

    point_proj = start + offset * start_to_end
    dist = geopy.distance.vincenty(tuple(point), tuple(point_proj)).meters
    # dist = float(np.linalg.norm(np.subtract(point, point_proj)))

    return dist


def create_contour(bounding_box, links, interval_length):
    """
        Creates a contour grid with contour positions selected by their
        distance from the links. Contours are created in intervals of
        interval_length.

    :param bounding_box:
    :param links:
    :param interval_length:
    :return: contour_dict = {distance: [receptors]}
    """
    step_lat = geopy.distance.vincenty(
        (bounding_box[0][0], bounding_box[0][1]),
        (bounding_box[1][0], bounding_box[0][1])).meters
    step_lon = geopy.distance.vincenty(
        (bounding_box[0][0], bounding_box[0][1]),
        (bounding_box[0][0], bounding_box[1][1])).meters
    # have contour not start on boundary, but slightly off
    eps = [(bounding_box[1][0] - bounding_box[0][0])
           / (10 * (step_lat / interval_length + 1)),
           (bounding_box[1][1] - bounding_box[0][1])
           / (10 * (step_lon / interval_length + 1))]
    list_lat = np.linspace(bounding_box[0][0] + eps[0],
                           bounding_box[1][0] - eps[0],
                           step_lat / interval_length + 1,
                           endpoint=True)
    list_lat = list_lat[1:len(list_lat)]
    list_lon = np.linspace(bounding_box[0][1] + eps[1],
                           bounding_box[1][1] - eps[1],
                           step_lon / interval_length + 1,
                           endpoint=True)
    list_lon = list_lon[1:len(list_lon)]
    # calculate minimal distance from vertices to line sources
    # store them in contour_dict {dist: [vertices]}
    contour_dict = defaultdict(list)
    for lat in list_lat:
        for lon in list_lon:
            dist = min([distance_point_from_line([lat, lon], link)
                        for link in links])
            # round to accuracy of interval
            dist = interval_length * int(dist / interval_length)
            contour_dict[dist].append([lat, lon])

    return contour_dict


def get_receptors(domain_dict, links_in_area,
                  contour_interval=25, include_stations=True):
    """
        Receptors are randomly allocated along contours spaced at
        contour_intervals intervals from the links. The receptors are
        spread such that four receptors are aligned to one contour if
        enough contours allow for that. If stations are included, then
        the furthest receptors are replaced by the positions of the
        stations in that area.

    :param domain_dict:
    :param links_in_area:
    :param contour_interval:
    :param include_stations:
    :return: receptors_dict = {sub_domain_id: [[lat, lon], ...]}
             receptors_dict_cart = {sub_domain_id: [[x, y], ...]}
    """
    stations = get_stations()

    receptors_dict = defaultdict(list)
    receptors_dict_cart = defaultdict(list)

    for sub_domain_id, sub_domain in domain_dict.items():
        bounding_box = sub_domain['coord']
        links = [links_in_area[link] for link in sub_domain['links']]

        # create contours at interval meter distance levels
        contour_dict = create_contour(bounding_box, links,
                                      interval_length=contour_interval)

        # continue if there are less than 20 possible receptor positions
        # available; this also removes this box from the domain_dict.
        contour_num = {dist: len(ct) for dist, ct in contour_dict.items()
                       if dist != 0}

        if include_stations:
            receptor_stations = [s for s in stations.values()
                                 if is_point_in_area(s, bounding_box)]
        else:
            receptor_stations = []
        num_receptors = 20 - len(receptor_stations)
        if sum(contour_num.values()) < num_receptors:
            continue

        contours = sorted(contour_dict.keys())
        # select contours to place receptors
        receptor_spacings = contours[1:min(6, len(contours))]
        # determine number or receptors to place on contours
        receptor_count = np.full(len(receptor_spacings), 0)
        while sum(receptor_count) < num_receptors:
            for i, spacing in enumerate(receptor_spacings):
                receptor_count[i] += (
                    1 if receptor_count[i] < contour_num[spacing] else 0
                )
                if sum(receptor_count) == num_receptors:
                    break

        for d, distance in enumerate(receptor_spacings):
            receptor_contour = contour_dict[distance]
            # indices = [int(i) for i in np.linspace(
            #   0, len(receptor_contour) - 1,
            #   min(receptor_count[d], len(receptor_contour))
            # )]
            indices = np.random.choice(range(len(receptor_contour)),
                                       receptor_count[d], replace=False)
            for index in indices:
                [r_lat, r_lon] = receptor_contour[index]
                receptors_dict[sub_domain_id].append([r_lat, r_lon])
                utm_coord = utm.from_latlon(r_lat, r_lon)
                receptors_dict_cart[sub_domain_id].append(list(utm_coord[:2]))
        # add stations
        for station in receptor_stations:
            [r_lat, r_lon] = station
            receptors_dict[sub_domain_id].append([r_lat, r_lon])
            utm_coord = utm.from_latlon(r_lat, r_lon)
            receptors_dict_cart[sub_domain_id].append(list(utm_coord[:2]))

        print(f'Receptors placed for sub-domain {sub_domain_id}.')

    return receptors_dict, receptors_dict_cart


def get_emitters(domain_dict, links_in_area):
    """
    :param domain_dict:
    :param links_in_area:
    :return: emitters_dict, emitters_dict_cart
    """
    emitters_dict = {key: [] for key in domain_dict.keys()}
    emitters_dict_cart = {key: [] for key in domain_dict.keys()}

    for sub_domain_id, sub_domain in domain_dict.items():
        for link in sub_domain['links']:
            lat_start = links_in_area[link][0][0]
            lat_end = links_in_area[link][1][0]
            lon_start = links_in_area[link][0][1]
            lon_end = links_in_area[link][1][1]
            utm_coord_start = utm.from_latlon(lat_start, lon_start)
            utm_coord_end = utm.from_latlon(lat_end, lon_end)

            emitters_dict[sub_domain_id].append(
                [lat_start, lon_start, lat_end, lon_end])
            emitters_dict_cart[sub_domain_id].append(
                list(utm_coord_start[:2] + utm_coord_end[:2]))

    return emitters_dict, emitters_dict_cart


def normalise(emitters_dict_cart, receptors_dict_cart):
    """
        normalize cartesian coordinates

    :param emitters_dict_cart:
    :param receptors_dict_cart:
    :return: norm_emitters, norm_receptors
    """
    # find maximum x and maximum y.
    x_min = min([v[0] for cart in emitters_dict_cart.values() for v in cart])
    y_min = min([v[1] for cart in emitters_dict_cart.values() for v in cart])
    # normalise
    norm_emitters = {
        key: [list(map(operator.sub, cart, [x_min, y_min, x_min, y_min]))
              for cart in emitters_cart]
        for key, emitters_cart in emitters_dict_cart.items()}
    norm_receptors = {key: [list(map(operator.sub, cart, [x_min, y_min]))
                            for cart in receptors_cart]
                      for key, receptors_cart in receptors_dict_cart.items()}

    return norm_emitters, norm_receptors


def get_utilities(collection_utilities, collection_traffic, ref_date,
                  bounding_box, contour_distance, include_stations, **kwargs):
    """
        Returns utility information from the database based on kwargs
        tags. If no information is available, a new dictionary is
        generated and added to the utilities database.

    :param collection_utilities:
    :param collection_traffic:
    :param ref_date: date of reference for which the traffic links are
    collected.
    :param bounding_box:
    :param contour_distance:
    :param include_stations:
    :param kwargs: identifier tags for Caline run
    :return:
    """
    print('Getting utilities ... ')
    util = uda.get_utilities_from_collection(collection_utilities,
                                             contour_distance=contour_distance,
                                             **kwargs)
    if len(util) > 0:
        return util
    else:
        print('Generating utilities:')
        if 'case' in kwargs and kwargs['case'] == 'Demo':
            util = create_demo(collection_utilities, contour_distance)
        else:  # 'case' == 'Dublin'
            # Generate Caline grids and associate link id's with grid numbers.
            traffic_sources = collection_traffic.find(
                {'date': ref_date.strftime('%Y-%m-%d %H:%M:%S')})
            if not traffic_sources.count():
                print(f'No traffic data available for starting date '
                      f'{ref_date} to generate link data. Please select '
                      f'an earlier or later date.')
                sys.exit(1)

            source_links = {}
            for source in traffic_sources:
                start = source['site']
                stop = source['next_site']
                if ((start, stop) not in source_links
                        and (stop, start) not in source_links):
                    source_links[(start, stop)] = [source['site_coord'],
                                                   source['next_site_coord']]

            util['bounding_box'] = bounding_box
            util['links_in_area'] = select_links_in_area(source_links,
                                                         bounding_box)
            print('Decompose domain ...')
            util['links_dict'], util['domain_dict'] = decompose_domain(
                bounding_box['coord_min'],
                bounding_box['coord_max'],
                util['links_in_area']
            )
            util['domain_neighbors'], util['intersections'] \
                = get_neighboring_domains(util['domain_dict'])
            print('Place receptors ... ')
            util['receptors_dict'], receptors_dict_cart = get_receptors(
                util['domain_dict'],
                util['links_in_area'],
                contour_distance,
                include_stations
            )
            util['receptors_index'] = dict()
            for sub_domain_id, receptor_list in util['receptors_dict'].items():
                for i, receptor in enumerate(receptor_list):
                    util['receptors_index'][tuple(receptor)] = {
                        'domain': sub_domain_id,
                        'index': i
                    }
            util['emitters_dict'], emitters_dict_cart = get_emitters(
                util['domain_dict'],
                util['links_in_area']
            )
            print('Normalize emitter and receptor coordinates.')
            util['norm_emitters'], util['norm_receptors'] = normalise(
                emitters_dict_cart,
                receptors_dict_cart
            )
            util['contour_distance'] = contour_distance
            util['stations'] = include_stations

        entry = uda.util_dict_to_db_entry(util)
        tags = {'contour_distance': contour_distance}
        for k, v in kwargs.items():
            entry[k] = v
            tags[k] = v

        print(f'Write utilities in database with tags: {tags}')
        collection_utilities.insert_one(entry)

    return util
