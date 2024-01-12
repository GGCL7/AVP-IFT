import os
import pandas as pd
import iFeatureOmegaCLI  # 确保已经安装了这个库
from DDE import *
def generate_features(input_txt_path):
    CKSAAGP = iFeatureOmegaCLI.iProtein(input_txt_path)
    CKSAAGP.get_descriptor("CKSAAGP type 2")

    PAAC = iFeatureOmegaCLI.iProtein(input_txt_path)
    PAAC.get_descriptor("PAAC")

    PAAC2 = iFeatureOmegaCLI.iProtein(input_txt_path)
    PAAC2.get_descriptor("PAAC")

    QSOrder = iFeatureOmegaCLI.iProtein(input_txt_path)
    QSOrder.get_descriptor("QSOrder")

    GTPC = iFeatureOmegaCLI.iProtein(input_txt_path)
    GTPC.get_descriptor("GTPC type 2")

    DistancePair = iFeatureOmegaCLI.iProtein(input_txt_path)
    DistancePair.get_descriptor("DistancePair")

    dde = feature_DDE(input_txt_path)  # 确保你有这个函数的定义

    # 重置索引
    PAAC.encodings = PAAC.encodings.reset_index(drop=True)
    DistancePair.encodings = DistancePair.encodings.reset_index(drop=True)
    CKSAAGP.encodings = CKSAAGP.encodings.reset_index(drop=True)
    GTPC.encodings = GTPC.encodings.reset_index(drop=True)
    QSOrder.encodings = QSOrder.encodings.reset_index(drop=True)
    dde = dde.reset_index(drop=True)

    # 合并所有的特征
    result = pd.concat([DistancePair.encodings, CKSAAGP.encodings, QSOrder.encodings, dde], axis=1)
    result.index = PAAC2.encodings.index
    result['Label'] = [1 if 'pos' in idx else 0 for idx in result.index]

    # 将Label列移动到第一列
    cols = result.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    result = result[cols]

    # 保存到csv文件
    return result

# # 使用函数
# x = generate_features('/Users/ggcl7/Desktop/AVP_data2/train1.txt')
# print(x)
# x.to_csv("/Users/ggcl7/Desktop/AVP_data2/feature1/train.csv", index=True, header=True, index_label="Id")