# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Function, Variable

# 得到对应的 scale_coef 和 dist_coef
def generate_coef(instance):
    diagonal_scale = math.sqrt(instance.size(0) ** 2 + instance.size(1) ** 2)
    # 统计共有多少个不同的 kernel 和每个位置对应几号 kernel
    unique_labels, unique_ids = torch.unique(instance, sorted=True, return_inverse=True)
    num_instance = unique_labels.size(0)
    if num_instance <= 1:
        return 0 
    pos_id = np.zeros((num_instance,4))
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        tmp = np.argwhere(instance == lb)
        print(tmp)
        pos_id[i,0] = tmp[0,0]
        pos_id[i,1] = tmp[1,0]
        pos_id[i,2] = tmp[0,-1]
        pos_id[i,3] = tmp[1,-1]

    coef = torch.zeros((num_instance,num_instance),dtype=torch.float32)
    for i,lb in enumerate(unique_labels):
        if lb == 0:
            continue
        for j,tb in enumerate(unique_labels):
            if j < i:
                continue
            elif j == i:
                coef[i,j] = math.exp(math.sqrt((pos_id[i,0]-pos_id[i,2]) ** 2 + (pos_id[i,1]-pos_id[i,3]) ** 2)/diagonal_scale/2)
            else:
                coef[i,j] = 1 - 20 * math.exp(-4-2.5*math.sqrt((pos_id[i,0]+pos_id[i,2]-pos_id[j,0]-pos_id[j,2]) ** 2 + (pos_id[i,1]+pos_id[i,3]-pos_id[j,1]-pos_id[j,3]) ** 2)/diagonal_scale)
                coef[j,i] = coef[i,j]
    return coef


class SA_loss(nn.Module):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)

    def forward_single(self, emb, instance, kernel, training_mask, bboxes):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)

        # 得到对应的Shape Aware 系数
        coef = generate_coef(instance)

        # 统计共有多少个不同的 kernel 和每个位置对应几号 kernel
        unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0

        # 计算属于同一 kernel 的 emb 平均值 
        emb_mean = emb.new_zeros((self.feature_dim, num_instance), dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

        # 对每个 instance 计算 aggregation loss，每一维度对应一个 instance
        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(coef[i,i] * dist - self.delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])

        # 计算 discrimination loss，将 instance 之间分离
        # 若 instance 只有 1 个（num_instance=2），loss 为 0
        if num_instance > 2:
            # repeat 是在对应维度将数据重复复制几次，permute 是将维度重新排列
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, self.feature_dim)
            # print(seg_band)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)

            coef_mask = (1 - torch.eye(num_instance, dtype=torch.int8))
            coef_mask[0,:] = 0
            coef_mask[:,0] = 0
            coef_dist = coef[coef_mask > 0].view(-1,1).reshape(1,-1)[0]

            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_d - coef_dist * dist) ** 2
            l_dis = torch.mean(torch.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        # 正则项，可能与聚类相关，待研究
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, bboxes, reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i], bboxes[i])

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch


