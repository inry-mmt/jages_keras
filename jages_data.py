# coding: utf-8

#
# Chainerで書いていたものをKerasで書き直す
# 出力 - 分類問題
#

from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import keras
import numpy as np

import pandas as pd

import random

class Network():
    def __init__(self):
        pass

    def constract(self, n_in, n_out):
        model = Sequential()

        model.add(Dense(4096, input_dim=n_in))
        model.add(LeakyReLU(alpha=0.25))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.25))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.25))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.25))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.25))
        model.add(BatchNormalization())
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.25))
        model.add(BatchNormalization())
        model.add(Dense(n_out))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        self.model = model

        return self.model

    def train(self, train_x, train_y, epoch=100, batch_size=16):
        early_stopping = EarlyStopping(monitor='loss',
                                       patience=16,
                                       mode='auto')

        callbacks = [early_stopping]

        self.model.fit(train_x, train_y, epochs=epoch, batch_size=batch_size,
                  callbacks=callbacks)

        return self.model

    def test(self, test_x, test_y):
        score = self.model.evaluate(test_x, test_y, batch_size=16)
        return score

    def save(self, FILEPATH="./model.h5"):
        self.model.save(FILEPATH)


def main():
    LABEL_FILE = "./labels.txt"
    CSV_FILE = "/home/bioinfo/ml/data/JAGES2013_data/result.csv"

    # ラベル情報取得
    with open(LABEL_FILE, "r") as f:
        labels = f.read().split('\n')

    # CSVファイル読み込み
    df = csv_to_dataframe(CSV_FILE)

    #　学習のターゲットにするやつ
    out_target = ["hhine_13"]
    # out_target = [labels[random.randint(0, len(labels) - 1)]]

    # xにあたる項目とyにあたる項目をそれぞれ用意
    in_labels = []
    out_labels = []
    for l in labels:
        if l in out_target:
            out_labels.append(l)
        else:
            in_labels.append(l)

    df = eliminate_uncomplete_rows(df, out_labels)


    # 学習に使えるフォーマットで取り出す
    x = extract_as_value(df, in_labels)
    y = extract_as_one_hot(df, out_labels)

    # 適当な比率で学習用とテスト用に分ける
    train_x = x[0:13000]
    train_y = y[0:13000]

    test_x = x[13000:]
    test_y = y[13000:]

    # 入出力次元数それぞれ　あまりきれいな取得ではない感じ
    n_in = train_x.shape[1]
    n_out = train_y.shape[1]

    # 学習モデルを作って学習する
    network = Network()
    network.constract(n_in, n_out)

    print("target - {}, start leargning".format(out_target))

    network.train(train_x, train_y, epoch=1000)

    # テスト
    score = network.test(test_x, test_y)

    print('\n\nおわり　スコア - {}'.format(score))

    # save
    network.save()

def csv_to_dataframe(filename):
    df = pd.read_csv(filename, header=0)
    return df

def equalize_by_category(data_x, data_y):
    """ not completed! """
    """
    data_y は one_hotな教師データ
    yの存在比率に合わせてdata_x, data_yを水増しする
    """

    num_each_category = {}
    for y in data_y:
        if y in num_each_category:
            num_each_category[y] += 1
        else:
            num_each_category[y] = 1



def eliminate_uncomplete_rows(df, reference_labels):
    """ reference_labels基準でNANな行を排除 """
    out_df_dict = {}

    uncomplete_row_numbers = []

    for label in reference_labels:
        for i, data in enumerate(df[label]):
            if np.isnan(data):
                uncomplete_row_numbers.append(i)

    for label in df:
        out_df_dict[label] = []

        for i, data in enumerate(df[label]):
            if i in uncomplete_row_numbers:
                pass
            else:
                out_df_dict[label].append(data)

    return pd.DataFrame.from_dict(out_df_dict)

def extract_as_one_hot(df, labels):
    out_df_list = []
    for l in labels:
        out_df_list.append(df[l])

    return keras.utils.to_categorical(np.array(out_df_list).T)

def extract_as_value(df, labels):
    out_df_list = []

    for l in labels:
        __df = df[l]
        __df = completion_by_mean(__df)
        __df = normalize(__df)

        out_df_list.append(__df)

    return np.array(out_df_list).T

def normalize(array):
    return array / np.max(array)

def completion_by_mean(array):
    """
    arrayのNanが含まれている列を全部平均値で埋めたやつを返す
    """

    out_df = {}

    dataset = list(array)
    mean = mean_without_nan(dataset)

    for i, d in enumerate(dataset):
        if np.isnan(d):
            dataset[i] = mean

    out_array = np.array(dataset)

    return out_array

def mean_without_nan(datalist):
    n = 0
    s = 0
    for d in datalist:
        if not np.isnan(d):
            n += 1
            s += d

    if n == 0:
        print("caution: no digits")
        return 0
    else:
        return s / n

if __name__ == "__main__":
    main()
