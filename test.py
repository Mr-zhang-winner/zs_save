import os

import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc

from model import RNN, DAtt, CSEN,T_UNet
from utils.BaseClass import ModelTrainer
from utils.TrainUtils import get_eval_index, get_pseudo_label, adjust_detection_res
from utils.dataset import DatasetFactory

g_count = 0

def ROC(point_label, cas):
    """
    计算ROC曲线
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(point_label, cas)
    # 计算每个阈值的TPR和FPR的差值
    diff = tpr - fpr

    # 找到最大差值对应的阈值
    optimal_threshold = thresholds[diff.argmax()]

    print('Optimal Threshold: ', optimal_threshold)

    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
class NetEvaluation:
    def __init__(self):
        # 初始化两个模型并将它们移动到GPU上
        self.model_CSEN = CSEN().cuda()
        self.model_DAtt = RNN().cuda()
        self.model_TUNet = T_UNet().cuda()

        # 从文件中加载预训练的模型权重

        # self.model_CSEN.load_state_dict(torch.load("./checkpoint-net1/Weak_Supervision/csen-100.pth"))
        # self.model_DAtt.load_state_dict(torch.load("./checkpoint-net1/Weak_Supervision/rnn-100.pth"))
        self.model_TUNet.load_state_dict(torch.load(r"E:\LJY\code_cy\cy_new\AnomalyDetection\训练2\newnewT_UNet_100.pth"))
        self.model_DAtt.load_state_dict(torch.load(r"E:\LJY\code_cy\cy_new\AnomalyDetection\训练2\newnewrnn_100.pth"))

        # 初始化数据集工厂，获取验证数据集，并创建数据加载器
        self.factory = DatasetFactory()
        self.val_dataset = self.factory.get_dataset("val")
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # 检测是否有可用的GPU，否则使用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化检测评价指标
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.acc = 0
        self.iou = 0
        self.fpr = 0
        # 初始化分类评价指标
        self.c_precision = 0
        self.c_recall = 0
        self.c_f1 = 0
        self.c_acc = 0
        self.c_iou = 0
        self.c_fpr = 0

    def set_zero(self):
        # 初始化检测评价指标
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.acc = 0
        self.iou = 0
        self.fpr = 0
        # 初始化分类评价指标
        self.c_precision = 0
        self.c_recall = 0
        self.c_f1 = 0
        self.c_acc = 0
        self.c_iou = 0
        self.c_fpr = 0

    def Evaluation(self):
        """
        计算并打印模型在验证集上的多个评价指标。
        """
        # 模型评估模式
        self.model_CSEN.eval()
        self.model_DAtt.eval()
        self.model_TUNet.eval()

        with (torch.no_grad()):
            # 遍历不同的beta值，beta用于调整两个模型输出的融合权重
            for i in range(0, 11):
                beta = 0.1 * i
                self.set_zero()
                labels_all = []
                cas_all = []
                # 遍历验证数据集中的每个批次
                for inputs, labels, point_label in self.val_loader:
                    # 将输入和标签移动到指定的设备（GPU或CPU）
                    inputs = inputs.to(self.device).float()
                    labels = labels.to(self.device).float()
                    labels = torch.argmax(labels, dim=1).float()  # 获取最可能的类别标签
                    point_label = point_label.to(self.device).float()

                    # 使用两个模型对输入进行预测
                    # out_csen, cas_csen = self.model_CSEN(inputs)
                    all_output_unet = self.model_TUNet(inputs)
                    pseudo_label_unet = all_output_unet[0]
                    p_class_unet = all_output_unet[1]
                    all_output_datt = self.model_DAtt(inputs)
                    pseudo_label_datt = all_output_datt[0]
                    p_class_datt = all_output_datt[1]
                    # out_datt, pseudo_label_datt = self.model_DAtt(inputs)

                    # 根据beta值融合两个模型的输出
                    # outputs = beta * out_csen + (1 - beta) * out_datt
                    # cas = beta * cas_csen + (1 - beta) * cas_datt
                    cas = beta * pseudo_label_unet + (1 - beta) * pseudo_label_datt
                    outputs = beta * p_class_unet + (1 - beta) * p_class_datt
                    labels_all.append(point_label.int().cpu().numpy().reshape(-1))
                    cas_all.append(cas.cpu().numpy().reshape(-1))
                    # 计算结果
                    # self.ObjEvaluation(labels, outputs, cas, point_label)

                    # 绘制主观结果
                    self.SubjEvaluation(inputs, point_label, pseudo_label_datt, pseudo_label_unet, cas, p_class_datt, p_class_unet, outputs, beta)
                labels_all = np.concatenate(labels_all)
                cas_all = np.concatenate(cas_all)
                ROC(labels_all, cas_all)
                self.ObjPrint(beta)  # 打印评价指标

    def ObjEvaluation(self, labels=None, outputs=None, cas=None, point_label=None):
        """
        计算模型在验证集上的多个评价指标
        """

        # 客观结果
        # 计算并累加检测评价指标
        p, r, f, a, i, tpr, fpr = get_eval_index(cas, point_label, outputs)
        self.precision += p
        self.recall += r
        self.f1 += f
        self.acc += a
        self.iou += i
        self.fpr += fpr

        # 计算并累加分类评价指标
        # 展平标签和预测结果
        labels_flat = np.squeeze(labels.cpu().numpy().reshape(1, -1))
        predictions_flat = (np.squeeze(outputs.cpu().numpy().reshape(1, -1)) > 0.5).astype(int)

        # 计算TP, FP, TN, FN
        TP = np.sum((labels_flat == 1) & (predictions_flat == 1))
        FP = np.sum((labels_flat == 0) & (predictions_flat == 1))
        TN = np.sum((labels_flat == 0) & (predictions_flat == 0))
        FN = np.sum((labels_flat == 1) & (predictions_flat == 0))

        # 计算指标
        self.c_precision += TP / (TP + FP) if (TP + FP) > 0 else 0
        self.c_recall += TP / (TP + FN) if (TP + FN) > 0 else 0
        self.c_acc += (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
        self.c_iou += TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        self.c_fpr += FP / (FP + TN) if (FP + TN) > 0 else 0
        pass

    def ObjPrint(self, beta):
        self.precision, self.recall, self.f1, self.acc, self.iou, self.fpr = (
            self.precision / len(self.val_loader),
            self.recall / len(self.val_loader),
            self.f1 / len(self.val_loader),
            self.acc / len(self.val_loader),
            self.iou / len(self.val_loader),
            self.fpr / len(self.val_loader),
        )
        self.c_precision, self.c_recall, self.c_acc, self.c_iou, self.c_fpr = (
            self.c_precision / len(self.val_loader),
            self.c_recall / len(self.val_loader),
            self.c_acc / len(self.val_loader),
            self.c_iou / len(self.val_loader),
            self.c_fpr / len(self.val_loader),
        )
        self.c_f1 += 2 * (self.c_precision * self.c_recall) / (self.c_precision + self.c_recall) \
            if (self.c_precision + self.c_recall) > 0 else 0
        # 打印每个beta值下的评价指标
        print(
            f"检测指标：  beta:{beta:.1f}, "
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, F1: {self.f1:.4f}, "
            f"ACC: {self.acc:.4f}, IOU: {self.iou:.4f}, FPR: {self.fpr:.4f}"
        )
        print(
            f"分类指标：  beta:{beta:.1f}, "
            f"Precision: {self.c_precision:.4f}, Recall: {self.c_recall:.4f}, F1: {self.c_f1:.4f}, "
            f"ACC: {self.c_acc:.4f}, IOU: {self.c_iou:.4f}, FPR: {self.c_fpr:.4f}"
        )
        print('**************************************************************************************')

    def SubjEvaluation(self, input, point_label, datt_cas, csen_cas, mix_cas, datt_cls, csen_cls, mix_cls, beta):
        """
        绘制主观结果
        :param input: 输入数据 (b,t,c)
        :param point_label: 点级标签 (b, t)
        :param datt_cas: DAtt模型的检测结果 (b, t)
        :param csen_cas: CSEN模型的检测结果 (b, t)
        :param mix_cas: 融合模型的检测结果 (b, t)
        :param datt_cls: DAtt模型的分类结果 (b,)
        :param csen_cls: CSEN模型的分类结果 (b,)
        :param mix_cls: 融合模型的分类结果 (b,)
        """

        def tensor_to_numpy(x):
            return x.cpu().numpy()

        input = tensor_to_numpy(input)
        point_label = tensor_to_numpy(point_label)
        datt_cas = tensor_to_numpy(datt_cas)
        csen_cas = tensor_to_numpy(csen_cas)
        mix_cas = tensor_to_numpy(mix_cas)
        datt_cls = tensor_to_numpy(datt_cls)
        # csen_cls = tensor_to_numpy(csen_cls)
        mix_cls = tensor_to_numpy(mix_cls)

        datt_pre_label, th_datt = get_pseudo_label(datt_cas, (datt_cls > 0.5).astype(np.int32))
        # csen_pre_label, th_csen = get_pseudo_label(csen_cas, (csen_cls > 0.5).astype(np.int32))

        th_datt = [0.5 for _ in range(8)]
        th_csen = [0.5 for _ in range(8)]
        datt_pre_label = datt_cas
        csen_pre_label = csen_cas

        mix_pre_label, th_mix = get_pseudo_label(mix_cas, (mix_cls > 0.5).astype(np.int32))
        # mix_pre_label = adjust_detection_res(point_label, mix_pre_label)
        # 绘制主观结果
        for b in range(input.shape[0]):
            global g_count
            g_count += 1
            input_batch = input[b, :, 2]
            point_label_batch = point_label[b]
            datt_cas_batch = datt_cas[b]
            csen_cas_batch = csen_cas[b]
            mix_cas_batch = mix_cas[b]
            datt_pre_label_batch = datt_pre_label[b]
            csen_pre_label_batch = csen_pre_label[b]
            th_mix_batch = th_mix[b]
            th_datt_batch = th_datt[b]
            th_csen_batch = th_csen[b]
            mix_pre_label_batch = mix_pre_label[b]

            # 定义变量
            fs = 12  # 字体大小
            ch_font = 'SimHei'  # 字体
            En_font = 'Times New Roman'  # 英文字体
            lw = 2  # 线宽

            # 创建一个新的figure，并指定子图的高度比例
            fig, axs = plt.subplots(8, 1, figsize=(10, 10))

            # 绘制input
            axs[0].plot(input_batch)
            axs[0].set_title('输入数据', fontproperties=ch_font, fontsize=fs)

            # 绘制point_label
            axs[1].plot(point_label_batch, '#f94144', linewidth=lw)
            axs[1].set_title('真值标签', fontproperties=ch_font, fontsize=fs)

            # 绘制datt_cas
            axs[2].plot(datt_cas_batch, '#f3722c', linewidth=lw)
            axs[2].plot(th_datt_batch * np.ones_like(datt_cas_batch), 'b--')
            # 在图中添加文本标签显示 th_mix 的值
            th_datt_formatted = "{:.4f}".format(th_datt_batch)  # 将 th_mix_batch 格式化为小数点后两位
            axs[2].text(0.2, th_datt_batch + 0.1, f'阈值={th_datt_formatted}', fontproperties=ch_font, fontsize=8,
                        color='blue')
            axs[2].set_title('Datt_cas', fontproperties=En_font, fontsize=fs)

            # 绘制csen_cas
            axs[3].plot(csen_cas_batch, '#f9c74f', linewidth=lw)
            axs[3].plot(th_csen_batch * np.ones_like(csen_cas_batch), 'b--')
            # 在图中添加文本标签显示 th_mix 的值
            th_csen_formatted = "{:.4f}".format(th_csen_batch)  # 将 th_mix_batch 格式化为小数点后两位
            axs[3].text(0.2, th_mix_batch + 0.1, f'阈值={th_csen_formatted}', fontproperties=ch_font, fontsize=8,
                        color='blue')
            axs[3].set_title('Csen_cas', fontproperties=En_font, fontsize=fs)

            # 绘制mix_cas
            axs[4].plot(mix_cas_batch, '#90be6d', linewidth=lw)
            axs[4].plot(th_mix_batch * np.ones_like(mix_cas_batch), 'b--')
            # 在图中添加文本标签显示 th_mix 的值
            th_mix_formatted = "{:.4f}".format(th_mix_batch)  # 将 th_mix_batch 格式化为小数点后两位
            axs[4].text(0.2, th_mix_batch + 0.1, f'阈值={th_mix_formatted}', fontproperties=ch_font, fontsize=8,
                        color='blue')
            axs[4].set_title('Cas', fontproperties=En_font, fontsize=fs)

            # 绘制datt_pre_label
            axs[5].plot(datt_pre_label_batch, '#43aa8b', linewidth=lw)
            axs[5].set_title('datt_pre_label', fontproperties=En_font, fontsize=fs)

            # 绘制csen_pre_label
            axs[6].plot(csen_pre_label_batch, '#213f8b', linewidth=lw)
            axs[6].set_title('csen_pre_label', fontproperties=En_font, fontsize=fs)

            # 绘制mix_pre_label
            axs[7].plot(mix_pre_label_batch, '#271f7f', linewidth=lw)
            axs[7].set_title('检测结果', fontproperties=ch_font, fontsize=fs)

            plt.tight_layout()
            plt.show()
            # 绘制完图表后保存到指定文件夹
            save_folder = "subjective_results_2"  # 保存结果的文件夹
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            file_name = f"beta{beta}batch_{b}_gcount_{g_count}.png"
            save_path = os.path.join(save_folder, file_name)
            plt.savefig(save_path)
            plt.close()
            # b-g_count


if __name__ == "__main__":
    net = NetEvaluation()
    net.Evaluation()
