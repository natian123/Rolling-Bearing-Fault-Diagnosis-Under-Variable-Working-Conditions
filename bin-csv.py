import struct
import numpy as np
import csv

# 假设每个样本有4个32位浮点数（即4列）
num_columns = 700
sample_size = 700 * num_columns  # 每个样本的字节数

# 打开 .bin 文件读取
with open('20240902_162621.bin', 'rb') as bin_file:
    data = []
    while True:
        # 读取一个样本（4个32位浮点数）
        sample = bin_file.read(sample_size)
        if len(sample) != sample_size:
            # 如果读取的数据不够完整的样本长度，跳出循环
            break
        # 使用 struct.unpack 将二进制数据解码为浮点数
        unpacked_data = struct.unpack('f' * num_columns, sample)
        data.append(unpacked_data)

    # 将数据转换为 NumPy 数组
    data_array = np.array(data)

    # 保存为 CSV 文件
    np.savetxt('output.csv', data_array, delimiter=',', fmt='%.6f')