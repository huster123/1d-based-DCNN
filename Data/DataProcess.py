import pandas as pd
import numpy as np

# 读取数据
raw1 = np.loadtxt("train_FD001.txt")
raw2 = np.loadtxt("test_FD001.txt")
raw3 = np.loadtxt("RUL_FD001.txt")
# 创建DataFrame
df1 = pd.DataFrame(raw1, columns=['unit', 'cycles', 'operational setting 1', 'operational setting 2',
                                  'operational setting 3'] + ['sensor measurement' + str(i) for i in range(1, 22)])
df2 = pd.DataFrame(raw2, columns=['unit', 'cycles', 'operational setting 1', 'operational setting 2',
                                  'operational setting 3'] + ['sensor measurement' + str(i) for i in range(1, 22)])
# 设定保留的传感器识数
indices = ['sensor measurement' + str(i) for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
# 剔除无意义的传感器识数
df1 = df1.loc[:, ['unit', 'cycles'] + indices]
df2 = df2.loc[:, ['unit', 'cycles'] + indices]
# 计算出训练集数据的平均值、标准差
means = [df1[i].mean() for i in indices]
stds = [df1[i].std() for i in indices]

label1 = []
label2 = []
# z-score标准化数据 & 提取出剩余寿命（RUL & label）
for i in range(df1.shape[0]):
    k = df1.loc[i, 'unit']
    m = df1.cycles[df1['unit'] == k].max()
    label1.append(m - df1.loc[i, 'cycles'] if (m - df1.loc[i, 'cycles']) < 125.0 else 125.0)
    for j in range(14):
        df1.iloc[i, j + 2] = (df1.iloc[i, j + 2] - means[j]) / stds[j]
for i in range(df2.shape[0]):
    k = df2.loc[i, 'unit']
    m = df2.cycles[df2.unit == k].max()
    label2.append(
        m - df2.loc[i, 'cycles'] + raw3[int(k - 1)] if (m - df2.loc[i, 'cycles'] + raw3[int(k - 1)]) < 125 else 125)
    for j in range(14):
        df2.iloc[i, j + 2] = (df2.iloc[i, j + 2] - means[j]) / stds[j]

df1['label'] = label1
df2['label'] = label2

slabel1 = []
slabel2 = []
unit1 = []
unit2 = []
valu1 = []
valu2 = []
# 构建时间序列数据
for i in range(df1.shape[0] - 29):
    if df1.loc[i, 'unit'] == df1.loc[i + 29, 'unit']:
        slabel1.append(df1.loc[i + 29, 'label'])
        unit1.append(df1.loc[i + 29, 'unit'])
        valu1.append(df1.iloc[i:i + 30, -15:-1].values.reshape(-1, ))
for i in range(df2.shape[0] - 29):
    if df2.loc[i, 'unit'] == df2.loc[i + 29, 'unit']:
        slabel2.append(df2.loc[i + 29, 'label'])
        unit2.append(df2.loc[i + 29, 'unit'])
        valu2.append(df2.iloc[i:i + 30, -15:-1].values.reshape(-1, ))
# 时间序列数据转化为DataFrame格式
valu1 = np.array(valu1)
valu2 = np.array(valu2)
df1 = pd.DataFrame(valu1)
df2 = pd.DataFrame(valu2)
df1['unit'] = unit1
df2['unit'] = unit2
df1['label'] = slabel1
df2['label'] = slabel2
# 保存处理后的数据
df1.to_csv("train_FD001.csv", index=0)
df2.to_csv("test_FD001.csv", index=0)

df1 = pd.read_csv('train_FD001.csv')
df2 = pd.read_csv('test_FD001.csv')

df1.to_pickle("train_FD001.pickle")
df2.to_pickle("test_FD001.pickle")
