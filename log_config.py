#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: log_config.py
@time: 2019-04-16 15:31
"""

import os
import datetime
import logging


class LogConfig:
    def __init__(self, log_type="console"):
        # 指定日志输出到控制台时的初始化
        if log_type == "console":
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(levelname)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                )
        # 指定日志输出到文件的初始化
        elif log_type == "file":
            # 创建存放日志的目录
            if not os.path.exists('./log'):
                os.mkdir('./log')

            # 操作系统本身不允许文件名包含:等特殊字符，所以这里也不要用，不然赋给filename时会报错
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d')

            file_name = "./log/'%s'.log" % (nowTime)
            file_handler = logging.FileHandler(filename=file_name, encoding='utf-8', mode='a')
            # level----指定打印的日志等级；默认为WARNING；可为NOTSET、DEBUG、INFO、WARNING、ERROR、CRITICAL
            # format----指定整条日志的格式；这里设置为“时间-等级-日志内容”
            # datefmt----format中时间的格式；
            # filename----日志输出到的文件；默认打印到控制台
            # filemode----日志文件读写形式；默认为“a”；配合filename使用，如果不用filename该参数也可不用
            # 本来输出到文件使用filename和filemode两个参数就可以了，不需要handlers
            # 但是logging将日志输出到文件时中文会乱码，而logging.basicConfig又没有提供指定编码的参数，要指定编码只能使用handlers
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(levelname)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                # filename=file_name,
                                # filemode='a',
                                handlers=[file_handler],
                                )

        # self.logger = logging.getLogger()

    def getLogger(self):
        logger = logging.getLogger()
        return logger
