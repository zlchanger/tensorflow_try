#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: EnglishFntDownload.py
@time: 2019-05-14 15:15
"""
import requests
from tqdm import tqdm
import os

# 下载数据集
fileurl = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz'
filename = 'EnglishFnt.tgz'
if not os.path.exists(filename):
    r = requests.get(fileurl, stream=True)
    with open(filename, 'wb') as f:
        for chunk in tqdm(r.iter_content(1024), unit='KB', total=int(r.headers['Content-Length'])/1024):
            f.write(chunk)