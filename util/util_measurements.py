import csv
import datetime

""" Utility methods for processing the measurement data.

Description:
    This module implements utility functions to import the supplemented measurement data
    from the .csv files 

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
    Julien Monteil <Julien.Monteil@ie.ibm.com>

Last updated:
    2019 - 05 - 04

"""


def get_stations():
    stations = dict()
    stations['Winetavern St Civic Offices'] = [53.344236, -6.271529]
    stations['Coleraine St. Housing Depot'] = [53.351600, -6.273196]
    stations['Ordinance Survey, Phoenix Park'] = [53.364562, -6.347643]
    stations['Rathmines Rd Upper'] = [53.322088, -6.267130]
    stations['Ballyfermot Library'] = [53.340006, -6.352282]
    stations['Marino Health Centre'] = [53.368179, -6.227809]
    stations['Finglas Civic Centre'] = [53.390295, -6.3045631]
    stations['Davitt Rd Waste Depot'] = [53.335605, -6.309578]
    stations['St Annes park – Red Stables'] = [53.359283, -6.175324]  # only available data from < 2015
    stations['Dun Laoghaire'] = [53.295753, -6.133805]  # ourside of bounding box
    stations['Blanchardstown'] = [53.384656, -6.379524]
    return stations


def get_empirical_background_pollution():
    """
         The empirical background pollution is taken to be the overall
         mean of the measurement data.
    :return: background_pollution
    """
    return {'NO2': 23.61 / 1881, 'PM25': 6.83, 'PM10': 11.38}


def annotate(site_location, coordinates, site):
    stations = get_stations()
    if 'Bally' in site:
        site_location.append('Ballyfermot Library')
        coordinates.append(stations['Ballyfermot Library'])
    elif 'Coleraine' in site:
        site_location.append('Coleraine St. Housing Depot')
        coordinates.append(stations['Coleraine St. Housing Depot'])
    elif 'Marino' in site:
        site_location.append('Marino Health Centre')
        coordinates.append(stations['Marino Health Centre'])
    elif 'Davitt' in site:
        site_location.append('Davitt Rd Waste Depot')
        coordinates.append(stations['Davitt Rd Waste Depot'])
    elif 'Phoenix' in site:
        site_location.append('Ordinance Survey, Phoenix Park')
        coordinates.append(stations['Ordinance Survey, Phoenix Park'])
    elif 'Finglas' in site:
        site_location.append('Finglas Civic Centre')
        coordinates.append(stations['Finglas Civic Centre'])
    elif 'Annes' in site:
        site_location.append('St Annes park – Red Stables')
        coordinates.append(stations['St Annes park – Red Stables'])
    elif 'Blanch' in site:
        site_location.append('Blanchardstown')
        coordinates.append(stations['Blanchardstown'])
    elif 'Lao' in site:
        site_location.append('Dun Laoghaire')
        coordinates.append(stations['Dun Laoghaire'])
    elif 'Winetavern' in site:
        site_location.append('Winetavern St Civic Offices')
        coordinates.append(stations['Winetavern St Civic Offices'])
    elif 'Wood' in site:
        site_location.append('Winetavern St Civic Offices')
        coordinates.append(stations['Winetavern St Civic Offices'])
    elif 'Rath' in site:
        site_location.append('Rathmines Rd Upper')
        coordinates.append(stations['Rathmines Rd Upper'])
    else:
        site_location.append(None)
        coordinates.append(None)
    return None


def str_to_float(string):
    if string.replace('.', '', 1).isdigit():
        return float(string)
    else:
        return -999


def flag_to_int(flag):
    if flag is '':
        return 9
    else:
        return int(float(flag))


def write_pm_csv_to_mongodb(file, collection):
    with open(file + '.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        row1 = next(csv_reader)
        sites = [row1[3 * index + 1] for index in range(int((len(row1) - 1) / 3))]
        pollutant_type = []
        site_location = []
        coordinates = []
        for site in sites:
            if '2.5' in site:
                pollutant_type.append('PM25')
            elif '10' in site:
                pollutant_type.append('PM10')
        for site in sites:
            annotate(site_location, coordinates, site)

        for row in csv_reader:
            for index, site in enumerate(sites):
                date = row[0]
                if coordinates[index] is None:
                    continue
                for hour in ['0' + s if len(s) == 1 else s for s in [str(n) for n in range(0, 24)]]:
                    current_date = datetime.datetime.strptime(date + ' ' + hour, '%d-%b-%y %H')
                    db_item = {'date': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                               'timestamp': current_date.timestamp(),
                               'site': site_location[index],
                               'pollutant': pollutant_type[index],
                               'coord': coordinates[index],
                               'value': str_to_float(row[3 * index + 1]),
                               'flag': flag_to_int(row[3 * index + 2])
                               }

                    collection.insert_one(db_item)
    return None


def write_nox_csv_to_mongodb(file, collection):
    with open(file + '.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        row1 = next(csv_reader)
        sites = row1
        site_location = []
        coordinates = []
        for site in sites:
            annotate(site_location, coordinates, site)

        _ = next(csv_reader)
        for row in csv_reader:
            for index, site in enumerate(site_location):

                date = row[0]
                time = row[1]
                if time[0:2] == '24':
                    time = '00' + time[2:5]
                current_date = datetime.datetime.strptime(date + ' ' + time, '%d/%m/%Y %H:%M')

                db_item = dict()
                db_item['date'] = current_date.strftime('%Y-%m-%d %H:%M:%S')
                db_item['timestamp'] = current_date.timestamp()
                db_item['site'] = site_location[index]
                db_item['coord'] = coordinates[index]
                db_item['flag'] = flag_to_int(row[6 * index + 6])

                db_item['pollutant'] = 'NO2'
                db_item['value'] = str_to_float(row[6 * index + 2])
                collection.insert_one(db_item)

                db_item['pollutant'] = 'NO'
                db_item['value'] = str_to_float(row[6 * index + 3])
                collection.insert_one(db_item)

                db_item['pollutant'] = 'NOx'
                db_item['value'] = str_to_float(row[6 * index + 4])
                collection.insert_one(db_item)
    return None
