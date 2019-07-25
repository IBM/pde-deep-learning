import copy
import pymongo
import numpy as np

import util.util_domain_decomposition as udd
import util.util_db_access as uda

""" Script for creating a two-tile demo example based on the collected data for Dublin. 

Description:
    This script takes the line source layout as was collected by a previous Caline run,
    and filters for the domains 6 and 7, which are in the Dublin city center. The positions of the
    line sources are adjusted such that each domain contains twenty lines, and that the lines
    are close to the domain boundary in domain 6, but far away in domain 7. The adjustment is done
    by hand.
    
    After that, the process of util.util_domain_decomposition.get_utilities() is followed to create
    the necessary utility information, including placement of receptors at various distances away 
    from the line sources. These utility objects are written into the database and ready to be
    used to run Caline.

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
    2019 - 05 - 21

"""


print("connecting to internal Mongo database ... ")
client_internal = pymongo.MongoClient('localhost', 27018)


def shift_links(contour_distance):
    run_tag = '2019-05-21 10 ' + str(contour_distance)
    tiles = [6, 7]

    collection_util = client_internal.db_air_quality.util
    util = uda.get_utilities_from_collection(collection_util, '2018-08-23 10 6')

    new_util = copy.deepcopy(util)
    del new_util['run_tag']  # add later

    coord_min = list(map(min, np.transpose([util['domain_dict'][t]['coord'][0] for t in tiles])))
    coord_max = list(map(max, np.transpose([util['domain_dict'][t]['coord'][1] for t in tiles])))
    new_util['bounding_box']['coord_min'] = coord_min
    new_util['bounding_box']['coord_max'] = coord_max
    new_util['domain_dict'] = {k: v for k, v in util['domain_dict'].items() if k in tiles}

    """ 
    select links in area 
    """
    # print(util['links_in_area'][(116, 120)])

    for link, coords in util['links_in_area'].items():
        if link == (67, 150):
            new_util['links_in_area'][link] = [[53.36189, -6.259736], [53.358773, -6.2651643]]
        if link == (9, 832):
            new_util['links_in_area'][link] = [[53.361427, -6.243033], [53.35844, -6.2419996]]
        if link == (1, 29):
            new_util['links_in_area'][link] = [[53.35484, -6.2488497], [53.352486, -6.2594585]]
        if link == (1, 796):
            new_util['links_in_area'][link] = [[53.35484, -6.2488497], [53.360167, -6.2505164]]
            new_util['domain_dict'][6]['links'].remove(link)
        if link == (559, 860):
            new_util['links_in_area'][link] = [[53.36006, -6.2465568], [53.35544, -6.245689]]
            new_util['domain_dict'][6]['links'].remove(link)
        if link == (861, 860):
            new_util['links_in_area'][link] = [[53.35404, -6.2410645], [53.35544, -6.245689]]
        if link == (660, 861):
            new_util['links_in_area'][link] = [[53.352713, -6.2412395], [53.35404, -6.2410645]]
            new_util['domain_dict'][6]['links'].remove(link)
        if link == (856, 861):
            new_util['links_in_area'][link] = [[53.35663, -6.241119], [53.35404, -6.2410645]]
        if link == (862, 861):
            new_util['links_in_area'][link] = [[53.354995, -6.2397747], [53.35404, -6.2410645]]
        if link == (862, 863):
            new_util['links_in_area'][link] = [[53.354995, -6.2397747], [53.360866, -6.2377973]]
        if link == (864, 865):
            new_util['links_in_area'][link] = [[53.359743, -6.2367663], [53.35977, -6.2383835]]
        if link == (248, 832):
            new_util['links_in_area'][link] = [[53.360846, -6.2449397], [53.36044, -6.2529996]]
            new_util['domain_dict'][7]['links'].append(link)
        if link == (72, 248):
            new_util['links_in_area'][link] = [[53.36006, -6.2460568], [53.360846, -6.2449397]]
            new_util['domain_dict'][7]['links'].append(link)
        if link == (865, 897):
            new_util['links_in_area'][link] = [[53.35977, -6.2383835], [53.360846, -6.2409397]]
            new_util['domain_dict'][7]['links'].append(link)
        if link == (911, 421):
            new_util['links_in_area'][link] = [[53.35977, -6.2653835], [53.355846, -6.2639397]]
            new_util['domain_dict'][7]['links'].append(link)
        if link == (173, 324):
            new_util['links_in_area'][link] = [[53.348153, -6.2655354], [53.347502, -6.2660896]]
        if link == (173, 622):
            new_util['links_in_area'][link] = [[53.348153, -6.2655354], [53.344813, -6.2657905]]
        if link == (305, 173):
            new_util['links_in_area'][link] = [[53.348185, -6.261613], [53.348153, -6.2655354]]
        if link == (49, 305):
            new_util['links_in_area'][link] = [[53.34428, -6.264448], [53.348185, -6.261613]]
        if link == (355, 17):
            new_util['links_in_area'][link] = [[53.348146, -6.251843], [53.348415, -6.255374]]
        if link == (439, 188):
            new_util['links_in_area'][link] = [[53.348223, -6.2488947], [53.345966, -6.2525]]
        if link == (75, 116):
            new_util['links_in_area'][link] = [[53.348362, -6.2457805], [53.346745, -6.2487785]]
        if link == (75, 254):
            new_util['links_in_area'][link] = [[53.348362, -6.2457805], [53.3478, -6.2417455]]
        if link == (868, 870):
            new_util['links_in_area'][link] = [[53.34512, -6.2382827], [53.34829, -6.2377713]]
        if link == (33, 157):
            new_util['links_in_area'][link] = [[53.34135, -6.26006], [53.342895, -6.2656275]]
            new_util['domain_dict'][6]['links'].append(link)
        if link == (7, 758):
            new_util['links_in_area'][link] = [[53.34788, -6.2455837], [53.347417, -6.2529044]]
            new_util['domain_dict'][6]['links'].append(link)
        if link == (116, 120):
            new_util['links_in_area'][link] = [[53.346745, -6.2487785], [53.345417, -6.2439044]]
            new_util['domain_dict'][6]['links'].append(link)

    new_util['links_in_area'] = {link: coords for link, coords in new_util['links_in_area'].items()
                                 if (link in new_util['domain_dict'][6]['links']
                                     or link in new_util['domain_dict'][7]['links'])}
    """
    decompose domain
    """
    new_util['links_dict'] = {}
    for tile, values in new_util['domain_dict'].items():
        for link in values['links']:
            new_util['links_dict'][tuple(link)] = [tile]

    """
    get neighboring domains
    """
    new_util['domain_neighbors'] = {k: [n for n in v if n in tiles]
                                    for k, v in util['domain_neighbors'].items() if k in tiles}
    new_util['intersections'] = {k: {kv: vv for kv, vv in v.items() if kv in tiles}
                                 for k, v in util['intersections'].items() if k in tiles}

    new_util['receptors_dict'], receptors_dict_cart = udd.get_receptors(new_util['domain_dict'],
                                                                        new_util['links_in_area'],
                                                                        contour_distance,
                                                                        False)
    new_util['receptors_index'] = dict()
    for sub_domain_id, receptor_list in util['receptors_dict'].items():
        for i, receptor in enumerate(receptor_list):
            new_util['receptors_index'][tuple(receptor)] = {'run_tag': run_tag,
                                                            'domain': sub_domain_id,
                                                            'index': i}

    new_util['emitters_dict'], emitters_dict_cart = udd.get_emitters(new_util['domain_dict'],
                                                                     new_util['links_in_area'])
    print('Normalize emitter and receptor coordinates.')
    new_util['norm_emitters'], new_util['norm_receptors'] = udd.normalise(emitters_dict_cart,
                                                                          receptors_dict_cart)

    for key, value in new_util.items():
        print(f'{key}: {value}')

    # borders = []
    # for tile, ngbrs in new_util["intersections"].items():
    #     for ngbr, border in ngbrs.items():
    #         borders.append(border)
    # fig = plt.figure(figsize=(7, 7))
    # # plot mesh grid
    # for line in borders:
    #     y = [line[0][0], line[1][0]]
    #     x = [line[0][1], line[1][1]]
    #     handle_b, = plt.plot(x, y, c='k')
    # for box_id, domain in new_util["domain_dict"].items():
    #     # plot line sources
    #     for link in domain['links']:
    #         [y, x] = np.transpose(new_util['links_in_area'][tuple(link)])
    #         handle_t, = plt.plot(x, y, c='b')
    #         plt.text(x[0], y[0], link[0])
    #         plt.text(x[1], y[1], link[1])
    #     # plot receptors
    #     for point in new_util["receptors_dict"][box_id]:
    #         y = point[0]
    #         x = point[1]
    #         handle_r = plt.scatter(x, y, c='r', marker='.')
    #
    # plt.show()

    # bson.errors.InvalidDocument: documents must have only string keys!
    entry = {key: ([(k, [(kk, vv) for kk, vv in v.items()] if key == 'intersections' else v)
                    for k, v in values.items()] if key != 'bounding_box' else values)
             for key, values in new_util.items()}
    entry['run_tag'] = run_tag

    print('')
    for key, value in entry.items():
        print(f'{key}: {value}')

    print('Write utilities in database.')
    collection_util.insert_one(entry)

    return None


if __name__ == '__main__':
    for dist in [5, 6, 7, 8, 9, 10, 11, 13, 17, 27, 37, 53, 71, 101, 103]:
        shift_links(dist)
