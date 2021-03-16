import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import time
from ..loss import build_loss, ohem_batch, iou
from ..post_processing.pa.pa import pa
from ..utils import Conv_BN_ReLU

# 参考 BiSeNet 提出的 self-attention 结构
class Self_Attention(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(Self_Attention,self).__init__()
        self.convblock = Conv_BN_ReLU(in_planes=self.in_channels, 
                                      out_planes=num_classes,
                                      stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class SA_Head(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_text,
                 loss_kernel,
                 loss_emb):
        super(PA_Head, self).__init__()
        self.block = Self_Attention(in_channels, num_classes)

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        return self.block(f)

    def get_results(self, out, img_meta, cfg):
        outputs = dict()

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :2, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        emb = out[:, 2:, :, :]
        emb = emb * text_mask.float()

        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        emb = emb.cpu().numpy()[0].astype(np.float32)

        # pa，输入的 kernels 和 emb 都是三维的
        label = pa(kernels, emb)

        # image size
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]

        label_num = np.max(label) + 1
        # 根据图像尺寸进行 resize
        label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_pa_time=time.time() - start
            ))

        scale = (float(org_img_size[1]) / float(img_size[1]),
                 float(org_img_size[0]) / float(img_size[0]))

        # 判断是否还需要 recognition(对于我们的实验无影响)
        with_rec = hasattr(cfg.model, 'recognition_head')

        if with_rec:
            bboxes_h = np.zeros((1, label_num, 4), dtype=np.int32)
            instances = [[]]

        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))

            # 小于一定大小的区域认为是误判，视作背景
            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue

            score_i = np.mean(score[ind])
            # 小于一定分数的区域认为是误判，视作背景
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue

            if with_rec:
                tl = np.min(points, axis=0)
                br = np.max(points, axis=0) + 1
                bboxes_h[0, i] = (tl[0], tl[1], br[0], br[1])
                instances[0].append(i)

            # 识别区域分为矩形 "rect" 和多边形区域 "poly" 分别处理，得到对应的 bbox 位置
            if cfg.test_cfg.bbox_type == 'rect':
                # minAreaRect 返回包含点集的最小矩形对象，boxPoints 返回这个矩形的四个边界点
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                # findContours 第一个返回值是一个 list，每个元素存储了可以表示一个轮廓的 ndarray
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)

        outputs.update(dict(
            bboxes=bboxes,
            scores=scores
        ))
        if with_rec:
            outputs.update(dict(
                label=label,
                bboxes_h=bboxes_h,
                instances=instances
            ))

        return outputs

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou(
            (kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        # embedding loss
        loss_emb = self.emb_loss(embs, gt_instances, gt_kernels[:, -1, :, :], training_masks, gt_bboxes, reduce=False)
        losses.update(dict(
            loss_emb=loss_emb
        ))

        return losses
