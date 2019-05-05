#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence
@file: DataLoader.py
@time: 2019-04-24 17:57
"""
import os, random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
from log_config import LogConfig

log_type = "file"
logger = LogConfig(log_type).getLogger()

class Sample:
    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        assert filePath[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.currIdxVal = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.valSamples = []
        self.maxTextLen = maxTextLen

        f = open(filePath + 'words.txt')
        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
        for line in f:
            # 忽略注解
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9

            # 路径格式: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + \
                       lineSplit[0] + '.png'

            # GT text are columns starting at 9
            gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
            chars = chars.union(set(list(gtText)))

            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            self.samples.append(Sample(gtText, fileName))

        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # 95%-train 5%-validation
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        self.numTrainSamplesPerEpoch = 25000
        self.numValSamplesPerEpoch = 5000

        # start with train set
        self.trainSet()
        self.validationSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))

    def truncateLabel(self, text, maxTextLen):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

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
            gtTexts = [self.samples[i].gtText for i in batchRange]

            X_data = np.ones([self.batchSize, self.imgSize[0], self.imgSize[1], 1])  # (bs, 128, 64, 1)
            Y_data = np.full([self.batchSize, self.maxTextLen], len(self.charList))
            input_length = np.ones((self.batchSize, 1))
            label_length = np.zeros((self.batchSize, 1))

            logger.info("-----------")
            for i in range(len(gtTexts)):
                img = preprocess(cv2.imread(self.samples[self.currIdx + i].filePath, cv2.IMREAD_GRAYSCALE),
                                 self.imgSize,
                                 self.dataAugmentation)
                X_data[i] = img
                word = self.make_target(gtTexts[i])
                Y_data[i, 0:len(word)] = word
                label_length[i] = len(word)
                input_length[i] = self.imgSize[0] // downsample_factor - 2

                msg = gtTexts[i]+'---'+self.samples[self.currIdx + i].filePath

                logger.info(msg)

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
            gtTexts = [self.valSamples[i].gtText for i in batchRange]

            X_data = np.ones([self.batchSize, self.imgSize[0], self.imgSize[1], 1])  # (bs, 128, 64, 1)
            Y_data = np.full([self.batchSize, self.maxTextLen], len(self.charList))
            input_length = np.ones((self.batchSize, 1))
            label_length = np.zeros((self.batchSize, 1))

            logger.info("-----------")
            for i in range(len(gtTexts)):
                img = preprocess(cv2.imread(self.valSamples[self.currIdxVal + i].filePath, cv2.IMREAD_GRAYSCALE),
                                 self.imgSize,
                                 self.dataAugmentation)
                X_data[i] = img
                word = self.make_target(gtTexts[i])
                Y_data[i, 0:len(word)] = word
                label_length[i] = len(word)
                input_length[i] = self.imgSize[0] // downsample_factor - 2

                logger.info(gtTexts[i], '---', self.valSamples[self.currIdxVal + i].filePath)

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


if __name__ == '__main__':
    filePath = '../data/'
    loader = DataLoader(filePath, 50, (128, 64), 32)
    loader.get_next_batch(4)
