# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-10-02 15:03:12
# Last modified: 2018-10-03 10:28:08

"""
Memory test.
"""

import numpy as np
import torch

# M = 500000
# 
# x = [1, 2, 3, 4, 5] * 100
# data = []
# 
# for i in range(M):
#     t = np.int32(x[:])
#     data.append(torch.from_numpy(t).long())
# 
# 
# print('I am done !')
# while True:
#     pass

x = [1, 2, 34]
x = np.int32(x)
y = torch.from_numpy(x).long()
print(y.type())
 
