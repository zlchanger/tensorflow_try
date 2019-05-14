# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence
@file: DataLoader.py
@time: 2019-05-14 17:57
"""
import os, random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from Logger import Logger

log = Logger('./log/info.log', level='info')

class DataLoader:

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        assert filePath[-1] == '/'

        self.filePath = filePath
        self.dataAugmentation = False
        self.currIdx = 0
        self.currIdxVal = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.valSamples = []
        self.trainSamples = []
        self.validationSamples = []
        self.maxTextLen = maxTextLen

        for i in range(10):
            path = self.filePath + 'Sample%03d/' % (i + 1)
            imgs = os.listdir(path)
            splitIdx = int(0.95 * len(imgs))
            self.trainSamples = self.trainSamples + imgs[:splitIdx]
            self.validationSamples = self.validationSamples + imgs[:splitIdx]
            self.samples = self.samples + imgs

        self.numTrainSamplesPerEpoch = 2000
        self.numValSamplesPerEpoch = 500

        # start with train set
        self.trainSet()
        self.validationSet()

        self.charList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def trainSet(self):
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

    def validationSet(self):
        self.dataAugmentation = False
        self.currIdxVal = 0
        random.shuffle(self.validationSamples)
        self.valSamples = self.validationSamples[:self.numValSamplesPerEpoch]

    def get_next_batch(self, downsample_factor):
        while True:
            if self.currIdx + self.batchSize > len(self.samples):
                self.trainSet()
            batchRange = range(self.currIdx, self.currIdx + self.batchSize)
            gtTexts = [get_value(self.samples[i]) for i in batchRange]
            # for i in batchRange:
            #     print(self.valSamples[i])
            #     print(get_value(self.valSamples[i]))

            X_data = np.ones([self.batchSize, self.imgSize[0], self.imgSize[1], 1])  # (bs, 128, 32, 1)
            Y_data = np.full([self.batchSize, self.maxTextLen], len(self.charList))
            input_length = np.ones((self.batchSize, 1))
            label_length = np.zeros((self.batchSize, 1))

            for i in range(len(gtTexts)):
                fileName = self.samples[self.currIdx + i]
                path = 'Sample' + fileName[3:6] + '/' + fileName
                img = preprocess(cv2.imread(self.filePath + path, cv2.IMREAD_GRAYSCALE), self.imgSize,
                                 self.dataAugmentation)
                X_data[i] = img
                word = self.make_target(gtTexts[i])
                Y_data[i, 0:len(word)] = word
                label_length[i] = len(word)
                input_length[i] = self.imgSize[0] // downsample_factor - 2

                msg = gtTexts[i] + '--' + path
                log.logger.info(msg)

            self.currIdx += self.batchSize

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batchSize])}
            yield (inputs, outputs)

    def get_val_next_batch(self, downsample_factor):
        while True:
            if self.currIdxVal + self.batchSize > len(self.valSamples):
                self.validationSet()
            batchRange = range(self.currIdxVal, self.currIdxVal + self.batchSize)
            gtTexts = [get_value(self.valSamples[i]) for i in batchRange]

            X_data = np.ones([self.batchSize, self.imgSize[0], self.imgSize[1], 1])  # (bs, 128, 64, 1)
            Y_data = np.full([self.batchSize, self.maxTextLen], len(self.charList))
            input_length = np.ones((self.batchSize, 1))
            label_length = np.zeros((self.batchSize, 1))

            for i in range(len(gtTexts)):
                fileName = self.valSamples[self.currIdxVal + i]
                path = 'Sample' + fileName[3:6] + '/' + fileName
                img = preprocess(cv2.imread(self.filePath+path, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
                X_data[i] = img
                word = self.make_target(gtTexts[i])
                Y_data[i, 0:len(word)] = word
                label_length[i] = len(word)
                input_length[i] = self.imgSize[0] // downsample_factor - 2

                if label_length[i] >= input_length[i]:
                    msg = gtTexts[i] + '---' + path
                    log.logger.info(msg)

            self.currIdxVal += self.batchSize

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batchSize])}
            yield (inputs, outputs)

    def make_target(self, text):
        return list(map(lambda x: self.charList.index(x), text))


def get_value(title):
    number = (int)(title[5])
    if number == 0:
        return '9'
    else:
        return (str)(number - 1)


if __name__ == '__main__':
    filePath = './English/Fnt'
    loader = DataLoader(filePath, 5, (128, 32), 10)
    loader.get_next_batch(4)
