import numpy as np
import torch
import cv2
from scipy.ndimage import label
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from torch.nn import init
import os as os


def get_pseudo_label(cas, cls_labels):
    """
    获取伪标签
    :param cas: cas结果
    :param cls_labels: 分类结果
    :return: 伪标签 , 阈值列表
    """
    th = []
    pseduo_label = np.zeros_like(cas)  # 初始化伪标签
    # 对cas中每个batch中的每个样本计算阈值
    for i in range(cas.shape[0]):
        # 单个batch的cas
        single_cas = cas[i]
        # 重塑以适应KMeans
        single_cas_reshaped = single_cas.reshape(-1, 1)
        # 应用KMeans聚类
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(single_cas_reshaped)
        # 计算两个聚类中心的平均值作为阈值
        threshold = float(np.mean(kmeans.cluster_centers_))
        th.append(threshold)
        # 使用确定的阈值生成伪标签
        single_pseudo_label = (cas[i, :] > threshold).astype(float) * float(cls_labels[i])
        # 更新伪标签
        pseduo_label[i] = single_pseudo_label
        # # 绘制cas值域分布图，并标在横轴上标注阈值
        # fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        #
        # # 子图1：显示single_cas的数据
        # axs[0].plot(single_cas, color='b')
        # # 在纵轴上标注阈值
        # axs[0].axhline(y=threshold, color='r', linestyle='--')
        # axs[0].text(0, threshold, 'Threshold: {:.2f}'.format(threshold), color='r')
        #
        # # 子图2：显示single_cas的分布
        # axs[1].hist(single_cas, bins=20, alpha=0.5, color='r')
        # # 在横轴上标注阈值
        # axs[1].axvline(x=threshold, color='r', linestyle='--')
        # axs[1].text(threshold, 0, 'Threshold: {:.2f}'.format(threshold), color='r')
        #
        # plt.tight_layout()
        # plt.show()

    return pseduo_label, th


def complete_cross_entropy(x, target):
    return -torch.mean(target * torch.log(x + 1e-6) + (1 - target) * torch.log(1 - x + 1e-6))


def get_learning_rate(e, initial_lr=0.001, decay_rate=0.5, decay_steps=30):
    """
    根据epoch获取学习率的函数
    Args:
        e: 当前的epoch数
        initial_lr: 初始学习率
        decay_rate: 学习率衰减率
        decay_steps: 学习率衰减的步数

    Returns:
        当前epoch对应的学习率
    """
    return initial_lr * decay_rate ** (e // decay_steps)


def get_eval_index(pseudo_label_pre, cas, point_label, outputs, th=0.5):
    """
    :param th:
    :param cas: b,t
    :param point_label:b,t
    :param outputs:b,2
    :return:
    """
    cas = cas.cpu().detach().numpy()  # (batch_size, t)
    point_label = (
        point_label.cpu().detach().numpy().astype(np.int32)
    )  # (batch_size, t)

    # outputs = outputs.cpu().detach().numpy().squeeze()
    # 二值化类别标签和cas
    # cls_labels = (outputs > th).astype(np.int32)  # 得到二值类别标签
    # cas_binary, _ = get_pseudo_label(cas, cls_labels)
    cas_binary = pseudo_label_pre.cpu().detach().numpy().astype(np.int32)
    # 调整预测结果
    # cas_binary = adjust_detection_res(point_label, cas_binary)

    # 展平标签和预测结果
    labels_flat = np.squeeze(point_label.reshape(1, -1))
    predictions_flat = np.squeeze(cas_binary.reshape(1, -1))

    # 计算TP, FP, TN, FN
    TP = np.sum((labels_flat == 1) & (predictions_flat == 1))
    FP = np.sum((labels_flat == 0) & (predictions_flat == 1))
    TN = np.sum((labels_flat == 0) & (predictions_flat == 0))
    FN = np.sum((labels_flat == 1) & (predictions_flat == 0))

    # 计算灵敏度(Sensitivity)和假阳性率(FPR)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    # 计算Precision
    precision = precision_score(
        np.squeeze(point_label.reshape(1, -1)),
        np.squeeze(cas_binary.reshape(1, -1)),
        average="binary",
        zero_division=1,
    )
    # 计算Recall
    recall = recall_score(
        np.squeeze(point_label.reshape(1, -1)),
        np.squeeze(cas_binary.reshape(1, -1)),
        average="binary",
        zero_division=1,
    )
    # 计算F1
    f1 = f1_score(
        np.squeeze(point_label.reshape(1, -1)),
        np.squeeze(cas_binary.reshape(1, -1)),
        average="binary",
        zero_division=1,
    )

    # 计算ACC
    acc = accuracy_score(
        np.squeeze(point_label.reshape(1, -1)),
        np.squeeze(cas_binary.reshape(1, -1)),
    )
    # 计算IOU
    iou = jaccard_score(
        np.squeeze(point_label.reshape(1, -1)),
        np.squeeze(cas_binary.reshape(1, -1)),
        average="binary",
    )

    return precision, recall, f1, acc, iou, sensitivity, fpr


def calculate_res(cas, label, k=0.1):
    cas = cas.squeeze(1)
    k = int(cas.shape[1] * k)
    # 沿着第二个维度找到前 k 个最大值及其索引
    _, topk_indices = torch.topk(cas, k, dim=1)
    # 使用索引提取前 k 个最大值
    topk_values = torch.gather(cas, 1, topk_indices)
    # 计算前 k 个最大值的平均值
    p_class = topk_values.mean(dim=1)

    # index = (label == 0)
    # p_class[index] = cas[index].mean(dim=1)
    return cas, p_class


import time
def process_cas(cas, label, save_folder=f'./output'):
    # 将 cas 转换为 numpy 数组以便进行图像处理
    cas_np = cas.cpu().detach().numpy()  # (8,1,17520)
    batch_size, channels, seq_len = cas_np.shape

    # 由于只有一个通道,直接squeeze掉最后一维
    cas_np = np.squeeze(cas_np, axis=1)  # (8,17520)
    # 二值化处理

    binary_map = (cas_np > 0.35).astype(np.uint8)

    # 形态学操作
    kernel = np.ones((60,), np.uint8)  # 使用更大的核以更好地连接相邻区域

    # 对每个batch分别处理
    nms_results = np.zeros((batch_size, seq_len))

    label = label.cpu().detach().numpy()

    for i in range(batch_size):

        # mean_center = np.mean(cas_np[i, :])
        # center = cas_np[i, :]
        # binary_map1 = (abs(center - mean_center) > 0.3).astype(np.uint8)
        # plt.plot(binary_map1)
        # plt.show()

        # fig, axs = plt.subplots(6, 1, figsize=(10, 15))
        # 绘制真值
        # axs[0].plot(label[i, :])
        # axs[0].set_title(f'real label (batch {i})')
        # # 绘制cas图
        # axs[1].plot(cas_np[i, :])
        # axs[1].set_title(f'Original CAS (batch {i})')
        # # 绘制对数处理图
        # axs[2].plot(cas_np[i, :])
        # axs[2].set_title(f'log (batch {i})')
        # # 绘制二值化后的图
        # axs[3].plot(binary_map[i, :])
        # axs[3].set_title(f'Binarized CAS (batch {i})')
        #
        # # 先进行膨胀操作,连接临近区域
        dilated = cv2.dilate(binary_map[i, :], kernel)

        # 再进行腐蚀操作,去除噪声
        morph_map = cv2.erode(dilated, kernel)


        # dilated = cv2.erode(binary_map[i, :], kernel)
        # morph_map = cv2.dilate(dilated, kernel)

        # # 绘制膨胀后的图
        # axs[4].plot(dilated)
        # axs[4].set_title(f'Dilated CAS (batch {i})')
        #
        # # 绘制腐蚀后的图
        # axs[5].plot(morph_map)
        # axs[5].set_title(f'Morphological CAS (batch {i})')
        #
        # # 寻找连通区域
        # # labeled_array, num_features = label(morph_map)
        #
        # # # 对每个连通区域进行NMS
        # # for j in range(1, num_features + 1):
        # #     region = (labeled_array == j)
        # #     if np.sum(region) > 0:
        # #         # 获取该区域的原始cas值
        # #         region_cas = np.expand_dims(cas_np[i, :], 1) * region
        # #         # 找出区域内最大响应值的位置
        # #         max_idx = np.argmax(region_cas)
        # #         # 在nms结果中只保留最大响应值位置
        # #         nms_results[i, max_idx] = 1
        nms_results[i, :] = morph_map.reshape(seq_len)

        # plt.tight_layout()
        # # # 保存图像
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        # save_path = os.path.join(save_folder, f'batch_{i}.png')
        # plt.savefig(save_path)
        # # plt.show()
        # plt.close()


    # out = np.expand_dims(nms_results, axis=2)  # 8 17520,1
    out = torch.from_numpy(nms_results).to(cas.device)
    out.requires_grad = True
    return out


# 新加的硬分类
def generate_pseudo_labels_zs(cas, label):
    # 生成伪标签的逻辑
    # 例如，使用阈值化生成二值序列
    # threshold = 0.5
    # outputs = np.squeeze(cas, axis=1)
    # pseudo_labels = (outputs > threshold).float()

    th = []
    cas_cpu = cas.cpu().detach().numpy()
    kmeans_labels = np.zeros_like(cas_cpu)  # 初始化伪标签
    kmeans_labels = np.squeeze(kmeans_labels, axis=1)

    label = label.cpu().detach().numpy()
    for i in range(cas.shape[0]):
        # plt.plot(label[i])
        # plt.show()

        # 单个batch的cas
        single_cas = cas_cpu[i]

        # 重塑以适应KMeans
        single_cas_reshaped = single_cas.reshape(-1, 1)
        # 应用KMeans聚类
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(single_cas_reshaped)

        difference = abs(kmeans.cluster_centers_[1].item() - kmeans.cluster_centers_[0].item())
        if difference > 0.2:
            # 计算两个聚类中心的平均值作为阈值
            threshold = float(np.mean(kmeans.cluster_centers_))
            th.append(threshold)
            # 使用确定的阈值生成伪标签
            single_pseudo_label = (cas_cpu[i, :] > threshold).astype(float)
            # 更新伪标签
            kmeans_labels[i] = single_pseudo_label
        # plt.plot(kmeans_labels[i])
        # plt.show()

        # 形态学操作
        # # 关门
        # # 先进行膨胀操作,连接临近区域
        # dilated = cv2.dilate(kmeans_labels[i, :], np.ones((60,), np.uint8))
        # # 再进行腐蚀操作,去除噪声
        # erode = cv2.erode(dilated, np.ones((60,), np.uint8))
        #
        # # 开门
        # erode = cv2.erode(erode, np.ones((20,), np.uint8))
        # dilated = cv2.dilate(erode, np.ones((20,), np.uint8))


        # 开门
        erode = cv2.erode(kmeans_labels[i, :], np.ones((5,), np.uint8))
        dilated = cv2.dilate(erode, np.ones((5,), np.uint8))

        # 关门
        # 先进行膨胀操作,连接临近区域
        dilated = cv2.dilate(dilated, np.ones((10,), np.uint8))
        # 再进行腐蚀操作,去除噪声
        erode = cv2.erode(dilated, np.ones((10,), np.uint8))



        kmeans_labels[i] = dilated.reshape(cas.shape[2])
        # plt.plot(kmeans_labels[i])
        # plt.show()
        # print('')
    pseudo_labels = torch.from_numpy(kmeans_labels).to(cas.device)
    pseudo_labels.requires_grad = True
    return pseudo_labels


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):  # 定义初始化函数，用于初始化网络的权重和偏置

        classname = m.__class__.__name__  # 获取当前层的类名

        # 对于包含权重的卷积层和线性层，根据不同的初始化类型进行初始化
        if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":  # 使用正态分布进行初始化
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":  # 使用xavier初始化方法
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":  # 使用kaiming初始化方法
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":  # 使用正交初始化方法
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )

            # 如果当前层包含偏置，并且偏置不为None，则将偏置初始化为0
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # 对于BatchNorm2d层，将其权重初始化为1，偏置初始化为0
        elif classname.find("BatchNorm1d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # 对网络应用初始化函数，实现权重和偏置的初始化


# 检测调整
def adjust_detection_res(g, p):
    """
    调整检测结果，将预测结果中连续的异常标记为异常
    :param g: 真实标签
    :param p: 预测标签
    :return: 调整后的预测标签
    """

    def adjust(gt, pred):
        anomaly_state = False
        # 遍历真实标签(gt)和预测标签(pred)
        for i in range(len(gt)):
            # 如果在某个位置，真实标签和预测标签都为1（表示异常），并且当前的异常状态为False
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                # 将异常状态设置为True
                anomaly_state = True

                # 从当前位置向前遍历
                for j in range(i, 0, -1):
                    # 如果在某个位置，真实标签为0（表示正常）
                    if gt[j] == 0:
                        # 则停止遍历
                        break
                    else:
                        # 否则，如果预测标签为0（表示正常）
                        if pred[j] == 0:
                            # 则将预测标签设置为1（表示异常）
                            pred[j] = 1

                # 从当前位置向后遍历
                for j in range(i, len(gt)):
                    # 如果在某个位置，真实标签为0（表示正常）
                    if gt[j] == 0:
                        # 则停止遍历
                        break
                    else:
                        # 否则，如果预测标签为0（表示正常）
                        if pred[j] == 0:
                            # 则将预测标签设置为1（表示异常）
                            pred[j] = 1

            # 如果在某个位置，真实标签为0（表示正常）
            elif gt[i] == 0:
                # 则将异常状态设置为False
                anomaly_state = False

            # 如果当前的异常状态为True
            if anomaly_state:
                # 则将当前位置的预测标签设置为1（表示异常）
                pred[i] = 1

    # 如果有batch维度，则遍历每个batch
    if len(g.shape) > 1:
        for i in range(g.shape[0]):
            adjust(g[i], p[i])
    return p
