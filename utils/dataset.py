import pickle

import numpy as np
import torch
import torch.utils.data as data


class DatasetFactory:
    """一个数据集工厂类，用来根据类型创建不同的数据集对象"""

    data = np.load(r'D:\ProgramData\pycharm_data\cy_new\AnomalyDetection\dataset-net1\data.pkl', allow_pickle=True)

    @classmethod
    def get_dataset(cls, dataset_type):
        """根据类型返回一个数据集对象"""
        if dataset_type == "train":
            return TrainDataset(data=cls.data)
        elif dataset_type == "val":
            return ValDataset(data=cls.data)
        else:
            return None


class TrainDataset(data.Dataset):
    """一个训练数据集类，继承自data.Dataset"""

    def __init__(self, data):
        self.data = data
        sorted_d = sorted(self.data.items(), key=lambda x: x[0])
        top_80_percent = int(len(sorted_d) * 0.8)
        self.data = dict(sorted_d[:top_80_percent])

    ...

    def __getitem__(self, index):
        x = self.data[index + 1]['data']  # T*C 维度
        y = self.data[index + 1]['label']  # 标签
        point_label = np.array(self.data[index + 1]['point_label'])  # 标签
        x = torch.from_numpy(x).cuda()
        y = torch.nn.functional.one_hot(torch.tensor(y), 2).cuda()
        point_label = torch.from_numpy(point_label).cuda()
        return x, y, point_label

    def __len__(self):
        return len(self.data)


class ValDataset(data.Dataset):
    """一个验证数据集类，继承自data.Dataset"""

    def __init__(self, data=None):
        self.data = data
        sorted_d = sorted(self.data.items(), key=lambda x: x[0])
        self.start_index = int(len(sorted_d) * 0.8)
        self.data = dict(sorted_d[self.start_index:])
        ...

    def __getitem__(self, index):
        index = self.start_index + index

        x = self.data[index + 1]['data']  # T*C 维度
        y = self.data[index + 1]['label']  # 标签
        point_label = np.array(self.data[index + 1]['point_label']).astype(float)  # 标签
        x = torch.from_numpy(x).cuda()
        y = torch.nn.functional.one_hot(torch.tensor(y), 2).cuda()
        point_label = torch.from_numpy(point_label).cuda()
        return x, y, point_label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # 使用工厂模式创建数据集对象
    factory = DatasetFactory()
    train_dataset = factory.get_dataset("train")
    val_dataset = factory.get_dataset("val")
    # 打印数据集的长度
    print(len(val_dataset))
    # 获取第一个样本
    sample = val_dataset[0]
    # x, y = sample
    # print(x)
    # print(y)

    # 遍历整个数据集
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        # x, y = sample
        # 进行其他操作
