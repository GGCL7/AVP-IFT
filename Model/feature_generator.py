import numpy as np
from BINARY import *
from BLOSUM62 import *
from ZSCALE import *
# , temp_file_path
from torch.utils.data import Dataset
import numpy as np



import torch
import numpy as np
import os

import torch
import numpy as np

import torch
import numpy as np


def feature_generator(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    fasta_list = np.array(f.readlines())

    datas = []  # 用于存储特征数据的列表
    labels = []  # 用于存储标签的列表
    sequences = []  # 用于存储序列的列表

    for flag in range(0, len(fasta_list), 2):
        fasta_str = [[fasta_list[flag].strip('\n').strip(), fasta_list[flag + 1].strip('\n').strip()]]
        sequences.append(fasta_str[0][1])  # 将序列添加到sequences列表

        bin_output = BINARY(fasta_str)
        blo_output = BLOSUM62(fasta_str)
        zsl_output = ZSCALE(fasta_str)

        if 'pos' in bin_output[1][0].split('>')[1]:
            feature_id = 1
        else:
            feature_id = 0
        labels.append(feature_id)  # 将标签添加到labels列表

        bin_output[1].remove(bin_output[1][0])
        blo_output[1].remove(blo_output[1][0])
        zsl_output[1].remove(zsl_output[1][0])

        bin_feature = []
        blo_feature = []
        zsl_feature = []

        for i in range(0, len(bin_output[1]), 20):
            temp = bin_output[1][i:i + 20]
            bin_feature.append(temp)
        for i in range(0, len(blo_output[1]), 20):
            temp = blo_output[1][i:i + 20]
            blo_feature.append(temp)
        for i in range(0, len(zsl_output[1]), 5):
            temp = zsl_output[1][i:i + 5]
            zsl_feature.append(temp)

        aa_fea_matrx = np.hstack([np.array(bin_feature), np.array(blo_feature), np.array(zsl_feature)])
        datas.append(aa_fea_matrx)  # 将特征矩阵添加到datas列表

    # 找到最大的行数
    max_rows = max(data.shape[0] for data in datas)

    # 将所有序列填充到最大的行数
    padded_datas = []
    for data in datas:
        if data.shape[0] <= max_rows:
            zeros = np.zeros((max_rows - data.shape[0], data.shape[1]))
            padded_data = np.vstack((data, zeros))
            padded_datas.append(padded_data)
    padded_datas_array = np.stack(padded_datas)
    padded_datas_array = padded_datas_array.astype(np.float32)

    return torch.tensor(padded_datas_array), torch.tensor(labels), sequences
    # return torch.tensor(padded_datas), torch.tensor(labels), sequences


if __name__ == '__main__':
    x, y, z= feature_generator('/Users/ggcl7/Desktop/AIPstack_data/data/train.txt')
    print(x)
