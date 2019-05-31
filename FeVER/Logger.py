# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Flyaway - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.4.5
#
# Date: 2017-03-07 11:57:32
# Last modified: 2019-04-05 09:54:27

"""
Config the logger for the system.

There are two different handlers for logger:
    - stream: logs information to the console, such as the loss and using time.
    - file: logs debug and information to the log file for further diagnosis.
"""

import logging

project_name = 'FeVER'


def initLogger(log_path):
    logger = logging.getLogger(project_name)
    logger.setLevel(logging.DEBUG)

    # Config the formatter
    formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s')

    # Config the stream handler
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # Config the file handler
    fileHandler = logging.FileHandler(log_path, mode='a')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
