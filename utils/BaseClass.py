# 导入抽象基类，用于定义模型训练的基类
from abc import ABC

import numpy as np
# 导入PyTorch库，用于深度学习模型的构建和训练
import torch
# 导入PyTorch数据工具包，用于数据加载和预处理
import torch.utils.data
# 导入PyTorch神经网络模块，用于定义模型结构
from torch import nn, optim
# 导入自定义的数据集工具类，用于创建和管理数据集
from utils.dataset import DatasetFactory
# 导入自定义的训练工具类，用于获取评估指标和学习率
from utils.TrainUtils import get_eval_index, get_learning_rate, complete_cross_entropy
from sklearn.cluster import KMeans
from utils.TIou import TIOULossCalculator, SequenceToBoxes
import time
# 定义模型训练基类
class ModelTrainer(ABC):
    def __init__(self):
        # 初始化评价指标

        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader = None, None, None, None
        self.epoch = 0
        self.acc = 0  # 准确率
        self.f1 = 0  # F1分数
        self.iou = 0  # 交并比
        self.fpr = 0
        self.tpr = 0
        self.recall = 0  # 召回率
        self.precision = 0  # 精确率
        self.val_loss = 0  # 验证集损失
        self.train_loss = 0  # 训练集损失
        self.best_f1 = 0  # 最佳F1分数
        self.epochs = 300  # 设置训练轮数
        self.save_path = ""  # 模型保存路径
        # 初始化优化器和模型
        self.optimizer = None
        self.model = None
        # 设置设备类型（GPU或CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化数据集工厂
        self.factory = DatasetFactory()  # 创建数据集工厂类实例
        # 加载数据集
        self.load_data()  # 调用数据加载函数

        # 初始化损失函数
        self.bce = None
        self.cross_entropy = complete_cross_entropy

        # new 2025 1 11
        self.converter = SequenceToBoxes()
        self.TIouLoss = TIOULossCalculator(iou_threshold=0.25)

    # 训练模型的函数
    def train_model(self):
        """
        使用类别标签训练RNN网络
        :return:
        """

        # 将模型设置为训练模式
        self.model.train()
        # 获取当前训练轮数对应的学习率
        lr = get_learning_rate(self.epoch)
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        # 遍历训练数据加载器中的每个批次
        i = 0
        pseudo_labels = []
        if self.epoch != 0:
            pseudo_labels_pre = torch.load(f'pseudo_labels.pt', weights_only=True)

        for inputs, labels, point_label in self.train_loader:
            # 将输入和标签移动到设备上
            print('\r', i+1, '   ', end="")
            i = i+1
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).float()
            # 使用torch.argmax获取最可能的类别标签
            labels = torch.argmax(labels, dim=1).float()
            # 将输入送入模型，得到输出
            pseudo_label, outputs, cas = self.model(inputs, point_label, labels)  # output---p_class
            # 将优化器的梯度清零
            self.optimizer.zero_grad()

            # 损失计算
            pseudo_labels.append(pseudo_label)
            if self.epoch == 0:
                # 计算输出和标签之间的损失
                loss = self.bce(outputs, labels)
            else:

                pseudo_label_pre_iter = pseudo_labels_pre[i-1].to(self.device)

                # 计算正样本和负样本的损失
                loss_pos = self.cross_entropy(cas[pseudo_label_pre_iter == 1], pseudo_label[pseudo_label_pre_iter == 1])
                loss_neg = self.cross_entropy(cas[pseudo_label_pre_iter == 0], pseudo_label[pseudo_label_pre_iter == 0])
                loss_detection = loss_pos + loss_neg
                # # 计算输出和标签之间的损失，包括BCE损失和交叉熵损失
                loss = self.bce(outputs, labels) + 0.01 * loss_detection

                # pseudo_label_1 = pseudo_label.flatten()
                # pseudo_label_pre_1 = pseudo_label_pre_iter.flatten()
                # preds = self.converter.sequence_to_segment(pseudo_label_pre_1)
                # boxes = self.converter.sequence_to_segment(pseudo_label_1)
                # TIou_loss = self.TIouLoss.calculate_loss(preds, boxes)
                # loss = self.bce(outputs, labels) + 0.01 * TIou_loss

            print(loss.item(), end="")
            # 反向传播损失，计算梯度
            loss.backward()
            # 更新优化器的参数
            self.optimizer.step()
            # 累加训练损失
            self.train_loss += loss.item()

        # 计算训练损失的平均值
        self.train_loss = self.train_loss / len(self.train_loader)
        print('   平均损失', self.train_loss)
        # 验证模型
        self.val_model()
        # 保存模型
        self.save_model()
        # 保存伪标签
        torch.save(pseudo_labels, f'pseudo_labels.pt')
        # 重置评价指标
        self.set_zero()


    # 使用伪标签训练模型的函数
    def train_model_with_pseudo_label(self, gt_model):
        # 将模型设置为训练模式
        self.model.train()
        # 获取当前训练轮数对应的学习率
        lr = get_learning_rate(self.epoch)
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        # 遍历训练数据加载器中的每个批次
        i=0
        for inputs, labels, point_label in self.train_loader:
            print('\r', i + 1, end="")
            i = i + 1
            # 将输入和标签移动到设备上
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).float()
            # 使用torch.argmax获取最可能的类别标签
            labels = torch.argmax(labels, dim=1).float()
            # 伪标签
            pseudo_label = gt_model.get_pseudo_label(inputs, point_label, labels)  # (b, t)
            # 将输入送入模型，得到输出
            pseudo_label_pre, outputs, cas = self.model(inputs)

            # pseudo_label_1 = pseudo_label.flatten()
            # pseudo_label_pre_1 = pseudo_label_pre.flatten()
            # preds = self.converter.sequence_to_segment(pseudo_label_pre_1)
            # boxes = self.converter.sequence_to_segment(pseudo_label_1)
            # loss = self.TIouLoss.calculate_loss(preds, boxes)

            # 计算正样本和负样本的损失
            loss_pos = self.cross_entropy(cas[pseudo_label == 1], pseudo_label[pseudo_label == 1])
            loss_neg = self.cross_entropy(cas[pseudo_label == 0], pseudo_label[pseudo_label == 0])
            loss_detection = loss_pos + loss_neg
            # # 计算输出和标签之间的损失，包括BCE损失和交叉熵损失
            loss = self.bce(outputs, labels) + 0.01 * loss_detection


            # 将优化器的梯度清零
            self.optimizer.zero_grad()
            # 反向传播损失，计算梯度
            loss.requires_grad_(True)
            loss.backward()
            # 更新优化器的参数
            self.optimizer.step()
            # 累加训练损失
            self.train_loss += loss.item()
        print('')
        # 计算训练损失的平均值
        self.train_loss = self.train_loss / len(self.train_loader)
        # 验证模型
        self.val_model(gt_model=gt_model)
        # 保存模型
        self.save_model()
        # 重置评价指标
        self.set_zero()

    def val_model(self, gt_model=None):
        # 将模型设置为评估模式
        self.model.eval()
        # 遍历验证数据加载器中的每个批次
        for inputs, labels, point_label in self.val_loader:
            # 将输入和标签移动到设备上
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).float()
            # 使用torch.argmax获取最可能的类别标签
            labels = torch.argmax(labels, dim=1).float()

            point_label = point_label.to(self.device).float()
            # 将输入送入模型，得到输出
            pseudo_label_pre, outputs, cas = self.model(inputs, point_label, labels)

            if gt_model is None:
                # 计算输出和标签之间的损失
                loss = self.bce(outputs, labels)
                # 累加验证损失
                self.val_loss += loss.item()
            else:
                # 伪标签
                # pseudo_label = gt_model.get_pseudo_label(inputs, labels)  # (b, t)
                pseudo_label = pseudo_label_pre
                # 计算正样本和负样本的损失
                loss_pos = self.cross_entropy(cas[pseudo_label == 1], pseudo_label[pseudo_label == 1])
                loss_neg = self.cross_entropy(cas[pseudo_label == 0], pseudo_label[pseudo_label == 0])
                loss_detection = loss_pos + loss_neg

                # pseudo_label_1 = pseudo_label.flatten()
                # pseudo_label_pre_1 = pseudo_label_pre.flatten()
                # preds = self.converter.sequence_to_segment(pseudo_label_pre_1)
                # boxes = self.converter.sequence_to_segment(pseudo_label_1)
                # loss_detection = self.TIouLoss.calculate_loss(preds, boxes)

                # 计算输出和标签之间的损失
                loss_cls = self.bce(outputs, labels)
                # 累加验证损失
                self.val_loss += (loss_cls.item() + 0.01 * loss_detection.item())

            # 累加评价指标
            p, r, f, a, i, tpr, fpr = get_eval_index(pseudo_label_pre, cas, point_label, outputs)
            self.precision += p
            self.recall += r
            self.f1 += f
            self.acc += a
            self.iou += i
            self.tpr += tpr
            self.fpr += fpr
        # 计算验证损失的平均值
        self.val_loss = self.val_loss / len(self.val_loader)
        # 计算评价指标的平均值
        self.precision, self.recall, self.f1, self.acc, self.iou, self.tpr, self.fpr = (
            self.precision / len(self.val_loader),
            self.recall / len(self.val_loader),
            self.f1 / len(self.val_loader),
            self.acc / len(self.val_loader),
            self.iou / len(self.val_loader),
            self.tpr / len(self.val_loader),
            self.fpr / len(self.val_loader),
        )
        # 打印每个训练轮的训练损失、验证损失和验证准确率
        print(
            f"Epoch {self.epoch + 1}/{self.epochs}, Train Loss: {self.train_loss:.4f}, Val Loss: {self.val_loss:.4f}, "
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}, "
            f"ACC: {self.acc:.4f}, IOU: {self.iou:.4f}, TPR: {self.tpr:.4f}, FPR: {self.fpr:.4f}"
        )

    def set_zero(self):
        # 重置评价指标为0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.acc = 0  # 准确率
        self.f1 = 0  # F1分数
        self.iou = 0  # 交并比
        self.fpr = 0
        self.tpr = 0
        self.recall = 0  # 召回率
        self.precision = 0  # 精确率

    # 加载数据集的函数
    def load_data(self):
        # 使用数据集工厂创建训练集和验证集
        self.train_dataset = self.factory.get_dataset("train")
        self.val_dataset = self.factory.get_dataset("val")
        # 创建训练集和验证集的数据加载器
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=8, shuffle=False, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 初始化模型的函数
    def initialize_model(self, optimizer=None):
        # 将模型移动到设备上
        self.model.to(self.device)
        # 定义损失函数和优化器
        self.bce = nn.BCELoss()
        self.cross_entropy = complete_cross_entropy
        # 如果没有提供优化器，则使用Adam优化器
        self.optimizer = (
            optim.Adam(self.model.parameters(), lr=0.001)
            if optimizer is None
            else optimizer
        )

    # 保存模型的函数
    def save_model(self):
        # 如果当前F1分数高于之前的最佳F1分数，则保存模型
        if self.f1 > self.best_f1:
            self.best_f1 = self.f1
            # 保存模型的状态字典到指定路径
            torch.save(self.model.state_dict(), self.save_path)

    def get_pseudo_label(self, input_data, label):
        """
        使用K均值聚类来确定动态阈值并生成伪标签。
        :param input_data: 输入的数据 (b, t, c)
        :param label: 类别标签 (b,)
        :return: 伪标签 (b, t)
        """

        self.model.eval()
        with torch.no_grad():
            # cls, cas = self.model(input_data)  # cls: (batch_size,) cas: (batch_size, t)
            pseudo_label, cls, cas = self.model(input_data)  # cls: (batch_size,) cas: (batch_size, t)
            # 初始化伪标签张量
        # pseudo_label = torch.zeros_like(cas)
        # # 遍历每个batch
        # for i in range(cas.shape[0]):
        #     # 单个batch的cas
        #     single_cas = cas[i].cpu().numpy()
        #     # 重塑以适应KMeans
        #     single_cas_reshaped = single_cas.reshape(-1, 1)
        #     # 应用KMeans聚类
        #     kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(single_cas_reshaped)
        #     # 计算两个聚类中心的平均值作为阈值
        #     threshold = float(np.mean(kmeans.cluster_centers_))
        #     # 使用确定的阈值生成伪标签
        #     single_pseudo_label = (cas[i] > threshold).float() * label[i].float()
        #     # 更新伪标签张量
        #     pseudo_label[i] = single_pseudo_label
        return pseudo_label
