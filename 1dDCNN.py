#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from matplotlib import pyplot as plt


def mean_squared_error(x, y):
    sum = 0
    n = len(x)
    for i, j in zip(x, y):
        sum = sum + (i - j) ** 2
    return sum / n


def score(x, y):
    sum = 0
    for i, j in zip(x, y):
        z = i - j
        if z < 0:
            sum = sum + np.e ** (-z / 13) - 1
        else:
            sum = sum + np.e ** (z / 10) - 1
    return sum


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def load_data():
    X = np.loadtxt("Data/train_FD001.txt")
    Y = np.loadtxt("Data/test_FD001.txt")
    Z = np.loadtxt("Data/RUL_FD001.txt")
    return X, Y, Z


def select_sensor(x, y):
    index = [0, 1, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
    new_x = x[:, index]
    new_y = y[:, index]
    print("筛选维度后的矩阵形状(包含编号)：", "训练集", new_x.shape, "测试集", new_y.shape)
    return new_x, new_y


def norm(x, y):
    for i in range(1, 15):
        a = np.mean(np.append(x[:, i + 1], y[:, i + 1]))
        b = np.std(np.append(x[:, i + 1], y[:, i + 1]))
        x[:, i + 1] = [(m - a) / b for m in x[:, i + 1]]  # z-score标准化
        y[:, i + 1] = [(m - a) / b for m in y[:, i + 1]]  # z-score标准化
    return x, y


def creat_dict(x, y):
    new_x = dict()
    new_y = dict()
    for line in x:
        if line[0] in new_x.keys():
            new_x[line[0]].append(line[1:])
        else:
            new_x[line[0]] = [line[1:]]
    for line in y:
        if line[0] in new_y.keys():
            new_y[line[0]].append(line[1:])
        else:
            new_y[line[0]] = [line[1:]]
    return new_x, new_y


def creat_rul(x, y, y_):
    rul_x = dict()
    rul_y = dict()
    for index in range(1, 101):
        rul_x[index] = [(x[index][-1][0] - m) if (x[index][-1][0] - m) < 125 else 125 for m in np.array(x[index])[:, 0]]
        rul_y[index] = [(y[index][-1][0] - m + y_[index - 1]) if (y[index][-1][0] - m + y_[index - 1]) < 125 else 125
                        for m in
                        np.array(y[index])[:, 0]]
    return rul_x, rul_y


def creat_image(x, y, x_, y_, all_y_):
    new_x = dict()
    new_y = dict()
    new_x_ = dict()
    new_all_y = dict()
    new_all_y_ = dict()
    for index in range(1, 101):
        new_x[index] = []
        new_y[index] = [m[1:] for m in y[index][-30:]]
        new_x_[index] = []
        new_all_y[index] = []
        new_all_y_[index] = []
        for i in range(0, len(x[index]) - 29):
            new_x[index].append([m[1:] for m in x[index][i:i + 30]])
            new_x_[index].append(x_[index][i + 29])
        for j in range(0, len(y[index]) - 29):
            new_all_y[index].append([m[1:] for m in y[index][j:j + 30]])
            new_all_y_[index].append(all_y_[index][j + 29])
    return new_x, new_y, new_x_, new_all_y, new_all_y_


def creat_data():
    x, y, y_ = load_data()
    x, y = select_sensor(x, y)
    x, y = norm(x, y)
    x, y = creat_dict(x, y)
    x_, all_y_ = creat_rul(x, y, y_)
    x, y, x_, all_y, all_y_ = creat_image(x, y, x_, y_, all_y_)
    a = []
    b = []
    a_ = []
    b_ = []
    b_ = y_
    for index in range(1, 101):
        a.extend([m for m in x[index]])
        b.append(y[index])
        a_.extend(x_[index])
    return x, y, x_, y_, a, b, a_, b_, all_y, all_y_


if __name__ == '__main__':
    x, y, x_, y_, input_x, input_y, output_x, output_y, all_y, all_y_ = creat_data()  # x, y, x_, y_,all_y,all_y_都是dict数据结构，其中all_y,all_y_是测试集的完全采集样本，用于检验单个编号发动机预测结果，样本数>100
    input_x = np.array(input_x)
    input_y = np.array(input_y)
    output_x = np.array(output_x)
    output_y = np.array(output_y)
    input_x = input_x.reshape(-1, 30, 14, 1)
    input_y = input_y.reshape(-1, 30, 14, 1)
    print(input_x.shape, input_y.shape)
    print(output_x.shape, output_y.shape)

    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh',
                     input_shape=(30, 14, 1)))
    # model.add(Dropout(0.5))
    # print('卷积1', model.output_shape)
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh'))
    # model.add(Dropout(0.5))
    # print('卷积2', model.output_shape)
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh'))
    # model.add(Dropout(0.5))
    # print('卷积3', model.output_shape)
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh'))
    # model.add(Dropout(0.5))
    # print('卷积4', model.output_shape)
    model.add(Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='tanh',
                     name='conv5'))  # 细节——第5层卷积层的卷积核为3*1，不是10*1
    model.add(Dropout(0.5))
    # print('卷积5', model.output_shape)
    model.add(Flatten())
    # print('平滑层', model.output_shape)
    model.add(Dropout(0.5))
    # print('dropout', model.output_shape)
    model.add(Dense(100, activation='tanh'))
    # print('连接层', model.output_shape)
    model.add(Dense(1, name='out'))
    # print('输出层', model.output_shape)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)
    history = model.fit(input_x, output_x, batch_size=512, epochs=200, shuffle=True)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)
    loss_image = model.fit(input_x, output_x, batch_size=512, epochs=50, shuffle=True)

    text_save("iteration.txt", np.append(np.array(history.history['loss']), np.array(loss_image.history['loss'])))

    plt.plot((history.history['loss']) + (loss_image.history['loss']), label='DCNN train 0~250')
    plt.legend(loc='upper right')
    plt.show()

    # model.save('DCNNmodel.h5')

    testPredict = model.predict(input_y)
    text_save("prediction result.txt", testPredict)
    RMSE = math.sqrt(mean_squared_error(output_y, testPredict))
    text_save("RMSE.txt", np.array([RMSE]))
    SCORE = score(testPredict, output_y)
    text_save("SCORE.txt", np.array([SCORE]))

    print("test rmse:", RMSE)
    print("test score:", SCORE)

    index = np.argsort(output_y)
    output_y = np.sort(output_y)
    testPredict = np.array([testPredict[i] for i in index])

    # calacu abs_error and rele_error
    AE = [abs(x - y) for x, y in zip(output_y, testPredict)]
    RE = [abs(x - y) / x for x, y in zip(output_y, testPredict)]

    # draw AE & RE & actual & prediction
    plt.figure(1, figsize=(15, 9))
    plt.subplot(221)
    plt.title('AE with increasing RUL')
    plt.plot(AE, 'r', lw=2, label='AE')
    plt.legend(loc='upper left')

    plt.subplot(222)
    plt.title('RE with increasing RUL')
    plt.plot(RE, 'r', lw=2, label='RE')
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.title('actual and prediction with increasing RUL')
    plt.plot(output_y, 'ob', ms=3)
    plt.plot(output_y, 'b', lw=2, label='actual')
    plt.legend(loc='upper left')
    plt.plot(testPredict, 'or', ms=3)
    plt.plot(testPredict, 'r', lw=2, label='prediction')
    plt.legend(loc='upper left')
    plt.show()

    # draw 21 & 24 & 34 & 81 unit actual & prediction
    plt.figure(1, figsize=(15, 9))
    test_all = [m for m in all_y[21]]
    test_all = np.array(test_all)
    test_all = test_all.reshape(-1, 30, 14, 1)
    test_all_predict = model.predict(test_all)
    plt.subplot(221)
    plt.title('test unit #21')
    plt.plot(all_y_[21], 'ob', ms=3)
    plt.plot(all_y_[21], 'b', lw=2, label='actual')
    plt.legend(loc='lower left')
    plt.plot(test_all_predict, 'or', ms=3)
    plt.plot(test_all_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='lower left')

    test_all = [m for m in all_y[24]]
    test_all = np.array(test_all)
    test_all = test_all.reshape(-1, 30, 14, 1)
    test_all_predict = model.predict(test_all)
    plt.subplot(222)
    plt.title('test unit #24')
    plt.plot(all_y_[24], 'ob', ms=3)
    plt.plot(all_y_[24], 'b', lw=2, label='actual')
    plt.legend(loc='lower left')
    plt.plot(test_all_predict, 'or', ms=3)
    plt.plot(test_all_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='lower left')

    test_all = [m for m in all_y[34]]
    test_all = np.array(test_all)
    test_all = test_all.reshape(-1, 30, 14, 1)
    test_all_predict = model.predict(test_all)
    plt.subplot(223)
    plt.title('test unit #34')
    plt.plot(all_y_[34], 'ob', ms=3)
    plt.plot(all_y_[34], 'b', lw=2, label='actual')
    plt.legend(loc='lower left')
    plt.plot(test_all_predict, 'or', ms=3)
    plt.plot(test_all_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='lower left')

    test_all = [m for m in all_y[81]]
    test_all = np.array(test_all)
    test_all = test_all.reshape(-1, 30, 14, 1)
    test_all_predict = model.predict(test_all)
    plt.subplot(224)
    plt.title('test unit #81')
    plt.plot(all_y_[81], 'ob', ms=3)
    plt.plot(all_y_[81], 'b', lw=2, label='actual')
    plt.legend(loc='lower left')
    plt.plot(test_all_predict, 'or', ms=3)
    plt.plot(test_all_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='lower left')
    plt.show()
