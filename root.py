# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-28 14:04:01
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-29 15:09:06

import platform

os_name = platform.system()
if os_name == 'Windows':
    dataroot = 'C:\\Users\\lemaire\\Documents\\SONIC paper data'
elif os_name == 'Linux':
    dataroot = '../../data/SONIC paper data/'
else:
    dataroot = None
