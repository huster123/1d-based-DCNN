import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import load_model


# 绘制制定unit的对比图象
def DrawUnit(df, No, model):
    test_input = df[df.unit == No].iloc[:, :-2].values.reshape((-1, 30, 14, 1))
    test_output = df[df.unit == No].iloc[:, -1].values.reshape(-1, )
    test_predict = model.predict(test_input)
    plt.figure(1, figsize=(15, 9))
    plt.title('test unit #' + str(No))
    plt.plot(test_output, 'ob', ms=3)
    plt.plot(test_output, 'b', lw=2, label='actual')
    plt.legend(loc='lower left')
    plt.plot(test_predict, 'or', ms=3)
    plt.plot(test_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    # 构建测试数据
    df = pd.read_pickle('Data/test_FD001.pickle')
    df1 = []
    df2 = []
    for i in range(100):
        df1.append(df[df.unit == i + 1].iloc[-1, :-2].values)
        df2.append(df[df.unit == i + 1].iloc[-1, -1])
    test_input = np.array(df1).reshape(-1, 30, 14, 1)
    test_output = np.array(df2).reshape(-1, )
    # 加载模型
    model = load_model('1dDCNNmodel.h5')
    # 得出预测结果
    test_predict = model.predict(test_input)
    # 绘制对比图象
    index = np.argsort(test_output)
    test_output = np.sort(test_output)
    test_predict = np.array([test_predict[i] for i in index])
    plt.figure(1, figsize=(15, 9))
    plt.title('actual and prediction with increasing RUL')
    plt.plot(test_output, 'ob', ms=3)
    plt.plot(test_output, 'b', lw=2, label='actual')
    plt.legend(loc='upper left')
    plt.plot(test_predict, 'or', ms=3)
    plt.plot(test_predict, 'r', lw=2, label='prediction')
    plt.legend(loc='upper left')
    plt.show()

    DrawUnit(df, 21, model)
    DrawUnit(df, 24, model)
    DrawUnit(df, 34, model)
    DrawUnit(df, 81, model)
