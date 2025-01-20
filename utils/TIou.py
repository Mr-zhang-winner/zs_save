#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/11 上午9:37
# @Author  : Mr Zhang
# @File    : DiouClass.py
# @Software: PyCharm


import numpy as np
import torch
import time
from utils.TrainUtils import process_cas

class TIOULossCalculator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def iou(self, box1, box2):
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.tensor(1)
        x2 = torch.min(box1[1], box2[1])
        y2 = torch.tensor(0)
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y1 - y2, min=0)
        area1 = (box1[1] - box1[0])
        area2 = (box2[1] - box2[0])
        union = area1 + area2 - intersection
        return intersection / union

    def center_distance(self, box1, box2):
        center1 = (box1[0] + box1[1]) / 2
        center2 = (box2[0] + box2[1]) / 2
        return torch.norm(center1 - center2)

    def diagonal_length(self, box1, box2):
        x1 = torch.min(box1[0], box2[0])
        x2 = torch.max(box1[1], box2[1])
        return torch.norm(torch.tensor([x2 - x1, torch.tensor(1)]))

    def diou(self, box1, box2):
        iou_val = self.iou(box1, box2)
        center_dist = self.center_distance(box1, box2)
        diagonal_len = self.diagonal_length(box1, box2)
        return iou_val - (center_dist ** 2) / (diagonal_len ** 2)

    def compute_diou_matrix(self, preds, boxes):
        num_preds = preds.size(0)
        num_boxes = boxes.size(0)
        diou_matrix = torch.zeros((num_preds, num_boxes))

        for i in range(num_preds):
            diou_matrix[i] = torch.stack([self.diou(preds[i], box) for box in boxes])
        return diou_matrix

        # for i in range(num_preds):
        #     for j in range(num_boxes):
        #         diou_matrix[i, j] = self.diou(preds[i], boxes[j])
        #         # diou_matrix[i, j] = self.iou(preds[i], boxes[j])
        # return diou_matrix

    def select_best_pairs(self, diou_matrix, preds, boxes):
        num_preds, num_boxes = diou_matrix.shape
        best_pairs = []
        used_preds = set()
        used_boxes = set()

        for _ in range(min(num_preds, num_boxes)):
            max_diou, max_idx = torch.max(diou_matrix.flatten(), dim=0)
            best_pred_idx, best_box_idx = divmod(max_idx.item(), num_boxes)

            if max_diou > 0 and self.iou(preds[best_pred_idx], boxes[best_box_idx]) > self.iou_threshold:
                best_pairs.append((best_pred_idx, best_box_idx))
                used_preds.add(best_pred_idx)
                used_boxes.add(best_box_idx)

            # Set used pairs to -1 to avoid re-selection
            diou_matrix[best_pred_idx, :] = -1
            diou_matrix[:, best_box_idx] = -1

        return best_pairs

    def calculate_loss(self, preds, boxes):
        if len(preds) == 0 and len(boxes) == 0:
            return torch.tensor(0.0)

        diou_matrix = self.compute_diou_matrix(preds, boxes)
        diou_matrix_copy = diou_matrix.clone().detach()
        best_pairs = self.select_best_pairs(diou_matrix_copy, preds, boxes)
        loss = torch.tensor(0.0)
        for pred_idx, box_idx in best_pairs:
            loss += 1 - diou_matrix[pred_idx, box_idx]
        loss += len(preds.clone().detach()) - len(best_pairs)
        return loss / len(preds.clone().detach())  # if best_pairs else 1.0


class SequenceToBoxes:
    def __init__(self):
        pass

    def sequence_to_segment(self, sequence):
        # 确保输入是一个一维张量
        sequence = sequence.squeeze()
        if len(sequence.shape) != 1:
            raise ValueError("Input tensor must be 1D")
        time_segment = []
        start = None
        sequence_cpu = sequence.cpu().detach().numpy()  # (8,1,17520)
        for i, bit in enumerate(sequence_cpu):
            if bit == 1:
                if start is None:
                    start = i
            elif start is not None:
                end = i
                time_segment.append([start, end])
                start = None
        # Handle the last segment if it ends with '1'
        if start is not None:
            end = len(sequence) - 1
            time_segment.append([start, end])
        boxes_tensor = torch.tensor(time_segment, dtype=torch.float32)
        return boxes_tensor


if __name__ == '__main__':

    # 示例数据
    # sequence1 = torch.tensor([[0, 1, 0, 1, 1, 1, 0, 0]])
    # # sequence1 = torch.tensor([[0,0, 0, 0, 0, 0, 0, 0]])
    # sequence2 = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0]])
    # # 创建SequenceToBoxes实例

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nms_results_pre = torch.rand((8, 1, 17520)).float().to(device)
    nms_results_rea = torch.rand((8, 1, 17520)).float().to(device)

    nms_results_pre = process_cas(nms_results_pre)
    nms_results_rea = process_cas(nms_results_rea)

    sequence1 = nms_results_pre.flatten()
    sequence2 = nms_results_rea.flatten()
    converter = SequenceToBoxes()
    # 转换序列
    preds = converter.sequence_to_segment(sequence1)
    segment = converter.sequence_to_segment(sequence2)
    print("Sequence 1 boxes:", preds)
    print("Sequence 2 boxes:", segment)

    # 示例数据
    # preds = [
    #     [10, 10, 50, 50],
    #     [20, 20, 60, 60],
    #     [30, 30, 70, 70],
    #     [40, 40, 80, 80],
    #     [50, 50, 90, 90]
    # ]
    #
    # boxes = [
    #     [15, 15, 55, 55],
    #     [25, 25, 65, 65],
    #     [35, 35, 75, 75],
    #     [45, 45, 85, 85],
    #     [55, 55, 95, 95],
    #     [65, 65, 105, 105]
    # ]

    # 创建DIOULossCalculator实例
    calculator = TIOULossCalculator(iou_threshold=0.35)

    # 计算DIoU矩阵
    diou_matrix = calculator.compute_diou_matrix(preds, segment)
    print("DIoU Matrix:")
    print(diou_matrix)

    # 选择最优框对
    best_pairs = calculator.select_best_pairs(diou_matrix, preds, segment)
    print("Best pairs (pred_idx, box_idx):")
    print(best_pairs)

    # 计算DIoU损失
    loss = calculator.calculate_loss(preds, segment)
    print("DIoU Loss:")
    print(loss)
