# This codes aims to process the Daily Mean Temperature.
# Please make sure the date satisfy the requirement.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from Simulation import simulation_temp

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

start_time = time.perf_counter()
# Read the temperature data, the file provide Beijing, Nanchang, Hohhot, Harbin, Tianjin, Ankang, Guangzhou, Shanghai,
# Xuzhou, Kunming, Liuzhou, Lanzhou, Shenyang, Jinan, Chengdu, Zhengzhou, Qiqihar, Karamay.
# In these data, there are not contain the Temp data of Feb.2nd in the Leap year.
# Because remove a data from a large data set doesn't make any influence to the whole distributed function of it.
# The read path is Relative Path.
# so when run this codes, please make sure the Folder of data 'Temp_data' and the main.py in the same Folder.
# 读取数据，包里共提供了北京、南昌、呼和浩特、哈尔滨、天津、安康、广州、上海、徐州、昆明、柳州、兰州、沈阳、济南、成都、郑州、齐齐哈尔、克拉玛依共18个地区的日平均气温，
# 这些气温数据不包含闰年2月29日的数据，在大量整体数据中删除某个数据不影响整体分布情况。 读取路径为相对路径，必须保证数据的文件夹Temp_data与主程序main.py在同一个路径下
Temp_data1 = pd.read_excel("Temp_data\\Beijing.xlsx")

# The unit of data read in is Fahrenheit scale, need to change into Celsius Scale.
# The first index of data is year, and second one is day after Transposed
# 读进来的数据单位为华氏度，需要换算为摄氏度
# 读取进来的数据转置后，第一索引为年份、第二索引为日期。
Temp_data = np.array((Temp_data1 - 32) * 5 / 9)
Temp_data_size = Temp_data.shape

# The size of figure is 15cm*10cm, and this size is suitable for single column in article.
# If possible, you can change the figure size following need.
# 图片大小为15cm*10cm的，这个尺寸的图片更适合单栏论文的排班，在实际使用中可以根据实际情况进行修改。
fig, ax = plt.subplots(figsize=(23.5, 10))

# Plot the scatter figure of the daily mean temperature.
# And in order the make sure the color of it will be same in every running, so we need to give a random seed.
# 绘制日平均气温的散点图，先给出一个随机数生成器
rnd = np.random
for j in range(Temp_data_size[1]):
    # The random seed will follow the year
    # If you want to make the color of the figure is different in every running, you can delete this sentence.
    # 根据年份给定一个随机种子，确保每次执行程序时得到的散点图颜色相同，如果想要每次散点图颜色不同，则可以将这句注释掉。
    rnd.seed(j)
    # This code make sure the color of scatter in the same year will be same.
    # 这个写法保证了每一年的日平均气温数据散点图颜色相同
    ax.scatter(np.array(range(365)), Temp_data[:, j], marker='o', facecolor='none',
               color=(rnd.random(), rnd.random(), rnd.random()))
# change the size
# 改变坐标轴字号大小
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# 设置横纵坐标轴标注
ax.set_xlabel('Time/ d', fontsize=18)
ax.set_ylabel('Daily Mean Temperature/ ℃', fontsize=18)
# 图名
ax.set_title('The Scatters of Daily Mean Temperature', fontsize=18)
plt.show()

# Now, we process the density function of the normal random process model.
# Calculating the mean and variation of Daily mean temperature
# 接下来处理密度拟合函数，计算每天日均气温数据的平均值和标准差
Temp_data = Temp_data.T
average_temp = np.mean(Temp_data, axis=0)
var_temp = np.sqrt(np.var(Temp_data, axis=0))

# The range is inside the 3 sigma of the highest and lowest temperature.
# This range will cover all the condition of data
# 密度拟合范围为最低、最高气温的3*sigma范围内，这个范围足够覆盖所有可能出现的情况
# 直接以最高最低气温作为边界也能够实现函数拟合
min_d = np.argmin(np.min(Temp_data))
max_d = np.argmax(np.max(Temp_data))
min_x = np.min(Temp_data) - 3 * var_temp[min_d]
max_x = np.max(Temp_data) + 3 * var_temp[max_d]
x = np.linspace(min_x, max_x, int((max_x - min_x) / 0.01))

# Calculating the density of every data.
# 计算密度值
f = np.zeros(np.size(x))
for j in range(Temp_data_size[0]):
    f = f + np.e ** (- ((x - average_temp[j]) ** 2) / (2 * (var_temp[j] ** 2))) / var_temp[j]
f = f / np.sqrt(2 * np.pi) / 365

# Calculating the density of normal distribution model in the code.
# 计算《相关规范》中建议模型的密度函数曲线
Temp_data_t = np.reshape(Temp_data, -1)
mean_Temp = np.mean(Temp_data_t)
var_Temp = np.sqrt(np.var(Temp_data_t))

f0 = np.e ** (- ((x - mean_Temp) ** 2) / (2 * (var_Temp ** 2))) / (
        np.sqrt(2 * np.pi) * var_Temp)

# plot three figure in the same figure.
# 将上述三个图像绘制在一起
fig = plt.figure(figsize=(15, 10))
ax1 = fig.subplots()
ax2 = ax1.twinx()

# 1. density function of normal random process.
# 1、 拟合密度函数曲线
ax2.plot(x, f, color='black', linewidth=3.0, label="the density function of normal random process model")
# 2. histogram of the real data.
# 2、绘制直方图，并将其与密度函数曲线绘制在一起
ax1.hist(Temp_data_t, bins=int(np.max(Temp_data) - np.min(Temp_data)), edgecolor='blue', color='steelblue',
         label="the histogram of the real data")
# 3. density function of normal distribution model.
# 3、《相关设计规范》建议模型的密度函数曲线
ax2.plot(x, f0, color='r', linewidth=3.0, label="the density function of normal distribution model")

# set the size of legend
# 设置坐标轴和图注文字大小
# 图注大小
ax1.legend(fontsize=18, bbox_to_anchor=(0.34, 0.88))
ax2.legend(fontsize=18, bbox_to_anchor=(0.55, 1))
# 坐标轴范围、坐标轴数字大小、坐标轴标注
ax2.set_ylim(0, np.max(f) * 1.2)
ax1.set_ylim(0, Temp_data_size[0] * Temp_data_size[1] * np.max(f) * 1.2)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax1.set_xlabel('Daily Mean Temperature/ ℃', fontsize=18)
ax1.set_ylabel('Day/ d', fontsize=18)
ax2.set_ylabel('Density', fontsize=18)
# 图名
ax1.set_title('The Histogram of Real Data, Density of Two Different Models', fontsize=18)
plt.show()

# 根据上述处理参数和模型生成模拟数据，不指定某天则默认生成全年数据，指定天数默认1月1日为第1天，依次类推，不包含闰年2月29日，每年共365天
# 必须指定生成多少年份数量的数据
# 函数输出数据为模拟生成数据，不指定则不返回相应数据
simulation_temp(average_temp, var_temp, 10)


# print the execution time of this algorithm
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"The number of data processed：{Temp_data_size[0] * Temp_data_size[1]} ")
print(f"The execution time is：{execution_time} seconds")
