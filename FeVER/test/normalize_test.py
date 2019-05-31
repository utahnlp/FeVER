# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2018-09-26 13:27:28
# Last modified: 2018-09-27 13:25:23

"""
Normalize
"""

import torch
import torch.nn as nn

weight = [[3, 4], [5, 12], [3, 6]]
weight = torch.FloatTensor(weight)
embed = nn.Embedding(3, 2)
weight = nn.Parameter(weight)
embed.weight = weight


print(embed.weight)

embed.weight.data = nn.functional.normalize(embed.weight.data)
print(embed.weight)
