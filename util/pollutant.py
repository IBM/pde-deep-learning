import enum
import math

"""

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
    Julien Monteil <Julien.Monteil@ie.ibm.com>

Last updated:
    2019 - 05 - 10
    
"""


class Pollutant(enum.Enum):
    NO2 = (46, 2.24, 2, 100, 0.006, 0.005)
    PM25 = (46, 0.1, 4, 96, 0.06, 0.1)
    PM10 = (46, 0.19, 4, 96, 0.06, 0.1)
    # CO = (30, 6.62, 1, 96)

    def get_name(self):
        return self.name

    def get_mol_weight(self):
        return self.value[0]

    def get_emission_factors(self, link_start, link_end):
        dist = math.sqrt((link_end[0] - link_start[0]) ** 2 + (link_end[1] - link_start[1]) ** 2) / 1000  # in km
        em_factor_link = self.value[1] * dist
        return em_factor_link

    def get_caline_number(self):
        return self.value[2]

    def get_line_number(self):
        return self.value[3]

    def get_ratio(self):
        return self.value[4]

    def get_max_decrease(self):
        return self.value[5]
