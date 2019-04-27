#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: tf_1_LSTM.py
@time: 2019-04-22 15:33
"""
import tensorflow as tf
from tensorflow import keras

import pandas as pd

mnist = keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(train_x.shape[1:]), activation=tf.nn.relu, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(128, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-6),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=3, batch_size=128, validation_data=(test_x, test_y))

df = pd.read_csv("crypto_data/LTC-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'])
print(df.head())

main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
for ratio in ratios:
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)
main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)
print(main_df.head())
