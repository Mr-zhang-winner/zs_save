# -*- coding: utf-8 -*-
# @Time    : 2023年8月8日11:11:41
# @Author  : cy
# 数据预处理脚本，将数据集转换为适合网络输入的格式
import os
import csv
import pickle

import pandas as pd

from utils.logger import LoggerFactory
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = LoggerFactory("data_process").get_logger()

folder_path = 'G:\\渗漏数据集\\Hanoi_CMH'


def read_csv_value(csv_file_path, value_csv):
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            value_csv.append(float(row[1]))


def list_subdirectories(folder_path):
    subdirectories = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if
                      os.path.isdir(os.path.join(folder_path, d))]
    return subdirectories


def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])


def StatisticsPressures(folder_path):
    """
    得到数据的 mean,var,scale
    :param folder_path: 数据集路径
    :return:
    """
    scenario_path = list_subdirectories(folder_path)

    # csv_list = ['Node_2.csv', 'Node_10.csv', 'Node_11.csv', 'Node_12.csv', 'Node_13.csv', 'Node_21.csv',
    #             'Node_22.csv', 'Node_23.csv', 'Node_31.csv', 'Node_32.csv']
    csv_list = [f'Node_{str(i)}.csv' for i in range(1, 33)]  # Hanoi有32个节点
    csv_list = sorted(csv_list, key=extract_number)
    logger.log_to_file("file,mean,var,scale")
    for i in csv_list:
        value_csv = []
        for s in scenario_path:
            read_csv_value(os.path.join(s, "Pressures", i), value_csv)
        value_csv = np.array(value_csv)
        scaler = StandardScaler()
        scaler.fit(value_csv.reshape(-1, 1))
        with open(f'../scalers/{i.split(".")[0]}-scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        logger.log_to_file(f"{i},{scaler.mean_[0]},{scaler.var_[0]},{scaler.scale_[0]}")
        print(f"[{i}:]均值：{scaler.mean_[0]} 方差：{scaler.var_[0]} 标准差：{scaler.scale_[0]}")



# def getData(folder_path):
#     """
#     获取训练数据{场景，数据，标签}
#     :param folder_path:
#     :return:
#     """
#     # 场景目录
#     scenario_path = list_subdirectories(folder_path)
#     # 数据文件
#     # csv_list = ['Node_2.csv', 'Node_10.csv', 'Node_11.csv', 'Node_12.csv', 'Node_13.csv', 'Node_21.csv',
#     #             'Node_22.csv', 'Node_23.csv', 'Node_31.csv', 'Node_32.csv']
#     csv_list = [f'Node_{str(i)}.csv' for i in range(1, 33)]  # Hanoi有32个节点
#
#     csv_list = sorted(csv_list, key=extract_number)
#     # 加载归一化器
#     scalers = {}
#     for root, dirs, files in os.walk("../scalers"):
#         for file in files:
#             if file.endswith('.pkl'):
#                 with open(os.path.join(root, file), 'rb') as f:
#                     scalers[file.split('-')[0]] = pickle.load(f)
#     # 构建数据
#     press = {}
#     with open(f"{folder_path}\\Labels.csv", 'r') as csvfile:
#         csvreader = csv.reader(csvfile)
#         next(csvreader)
#         labels = list(csvreader)
#     for s in scenario_path:  # 场景
#         press[int(s.split('-')[1])] = {'data': [], 'label': -1}
#         for i in csv_list:  # 数据
#             value_csv = []
#             read_csv_value(os.path.join(s, "Pressures", i), value_csv)
#             # 调整数据
#             tem = []
#             for j in value_csv:
#                 v = scalers[i.split('.')[0]].transform([[j]])[0][0]
#                 tem.append(v)
#             press[int(s.split('-')[1])]['data'].append(tem)
#         press[int(s.split('-')[1])]['data'] = np.array(press[int(s.split('-')[1])]['data']).T
#         ...
#         press[int(s.split('-')[1])]['label'] = int(float(labels[int(s.split('-')[1]) - 1][1]))
#         print(f"场景：{int(s.split('-')[1])}")
#     f = open('../dataset/data.pkl', 'wb')
#     pickle.dump(press, f)


def getData(folder_path):
    scenario_path = list_subdirectories(folder_path)
    csv_list = [f'Node_{str(i)}.csv' for i in range(1, 33)]
    csv_list = sorted(csv_list, key=extract_number)

    # 预加载所有的归一化器
    scalers = {file.split('-')[0]: pickle.load(open(os.path.join(root, file), 'rb'))
               for root, dirs, files in os.walk("../scalers") for file in files if file.endswith('.pkl')}

    press = {}
    scenario_path = sorted(scenario_path, key=lambda x: int(x.split('Scenario-')[-1]))  # 排序

    for s in scenario_path:
        sc_num = int(s.split('-')[1])  # 场景编号
        press[sc_num] = {'data': [], 'label': -1, 'point_label': None}
        for i in csv_list:  # 各个节点压力数据
            value_csv = pd.read_csv(os.path.join(s, "Pressures", i)).values[:, 1].reshape(-1, 1)
            tem = scalers[i.split('.')[0]].transform(value_csv).flatten()
            press[sc_num]['data'].append(tem)
        press[sc_num]['data'] = np.array(press[sc_num]['data']).T
        # 点级标签
        press[sc_num]['point_label'] = pd.read_csv(os.path.join(s, "Labels.csv", )).values[:, 1]

        press[sc_num]['label'] = 1 if press[sc_num]['point_label'].sum() > 0 else 0
        print(f"场景：{sc_num}")

    with open('../dataset/data.pkl', 'wb') as f:
        pickle.dump(press, f)


if __name__ == "__main__":
    # StatisticsPressures(folder_path)
    getData(folder_path)
