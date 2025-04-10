import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('外圈轻度实测800rpm(扩充).csv')

# 获取数据列
time_series_data = data.iloc[:, 0]

# 绘制时序波形图
plt.plot(time_series_data)

# 添加标题和轴标签
plt.title('Time Series Waveform')
plt.xlabel('Time')
plt.ylabel('Data')

# 显示图形
plt.show()