import numpy as np
import cv2
import torch
cimport numpy as np
cimport cython
cimport libcpp
cimport libcpp.pair
cimport libcpp.queue
from libcpp.pair cimport *
from libcpp.queue  cimport *

@cython.boundscheck(False)
@cython.wraparound(False)

cdef np.ndarray[np.int32_t, ndim=2] _pa(np.ndarray[np.uint8_t, ndim=3] kernels,
                                        np.ndarray[np.float32_t, ndim=3] emb,
                                        np.ndarray[np.int32_t, ndim=2] label,
                                        np.ndarray[np.int32_t, ndim=2] cc,
                                        int kernel_num,
                                        int label_num,
                                        float min_area=0):
    # kernels 是 output 中 kernels 的位置，1*h*w
    # emb 是 output 中的 emb，4*h*w
    # label 是通过 kernels 连通分量得到的不同连通标记，h*w
    # cc 是通过 text_mask 连通分量得到的不同连通标记，h*w
    # kernel_num = 2，在此设定下
    # label_num 是 label 中的不同连通分量个数
    cdef np.ndarray[np.int32_t, ndim=2] pred = np.zeros((label.shape[0], label.shape[1]), dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] mean_emb = np.zeros((label_num, 4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] area = np.full((label_num,), -1, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] flag = np.zeros((label_num,), dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=3] inds = np.zeros((label_num, label.shape[0], label.shape[1]), dtype=np.uint8)
    cdef np.ndarray[np.int32_t, ndim=2] p = np.zeros((label_num, 2), dtype=np.int32)

    cdef np.float32_t max_rate = 1024
    # inds 维度是 label_num*h*w，分别存储了对应 kernel 的二值化图
    # area 维度是 label_num，分别存储了对应 kernel 的区域大小，若小于一定值则抛弃，改写 label
    # p 维度是 label_num*2，每行存储的是左上角像素的坐标
    for i in range(1, label_num):
        ind = label == i
        inds[i] = ind

        area[i] = np.sum(ind)

        if area[i] < min_area:
            label[ind] = 0
            continue

        px, py = np.where(ind)
        p[i] = (px[0], py[0])

        for j in range(1, i):
            if area[j] < min_area:
                continue
            # 仅有在 text_mask 中不同 kernel 对应的 instance 存在纠缠现象时需要考虑
            if cc[p[i, 0], p[i, 1]] != cc[p[j, 0], p[j, 1]]:
                continue
            rate = area[i] / area[j]
            if rate < 1 / max_rate or rate > max_rate:
                flag[i] = 1
                mean_emb[i] = np.mean(emb[:, ind], axis=1)

                if flag[j] == 0:
                    flag[j] = 1
                    mean_emb[j] = np.mean(emb[:, inds[j].astype(np.bool)], axis=1)

    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    cdef libcpp.queue.queue[libcpp.pair.pair[np.int16_t, np.int16_t]] nxt_que = \
        queue[libcpp.pair.pair[np.int16_t, np.int16_t]]()
    cdef np.int16_t*dx = [-1, 1, 0, 0]
    cdef np.int16_t*dy = [0, 0, -1, 1]
    cdef np.int16_t tmpx, tmpy

    # 把 kernel 上的 text instance 标记到 pred 上 
    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        tmpx, tmpy = points[point_idx, 0], points[point_idx, 1]
        que.push(pair[np.int16_t, np.int16_t](tmpx, tmpy))
        pred[tmpx, tmpy] = label[tmpx, tmpy]

    cdef libcpp.pair.pair[np.int16_t, np.int16_t] cur
    cdef int cur_label
    for kernel_idx in range(kernel_num - 2, -1, -1):
        while not que.empty():
            cur = que.front()
            que.pop()
            cur_label = pred[cur.first, cur.second]

            is_edge = True
            # 实际上就是不断的将 kernel 旁边的像素扩展到 kernel 中（通过入栈出栈实现），其中遵循了一定的原则
            for j in range(4):
                tmpx = cur.first + dx[j]
                tmpy = cur.second + dy[j]
                # 原则一：超出边界的点不算
                if tmpx < 0 or tmpx >= label.shape[0] or tmpy < 0 or tmpy >= label.shape[1]:
                    continue
                # 原则二：先到先得，已被标记的或者不算 text 的不添加
                if kernels[kernel_idx, tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                # 原则三：聚类不够相似的不算（emb 距离）
                if flag[cur_label] == 1 and np.linalg.norm(emb[:, tmpx, tmpy] - mean_emb[cur_label]) > 3:
                    continue
                    
                que.push(pair[np.int16_t, np.int16_t](tmpx, tmpy))
                pred[tmpx, tmpy] = cur_label
                is_edge = False
            if is_edge:
                nxt_que.push(cur)

        que, nxt_que = nxt_que, que

    return pred

def pa(kernels, emb, min_area=0):
    kernel_num = kernels.shape[0]
    # 分别对于 text_mask 和 kernel 去求连通分量，将相关内容输入到 _pa()函数得到最终的 label
    _, cc = cv2.connectedComponents(kernels[0], connectivity=4)
    label_num, label = cv2.connectedComponents(kernels[1], connectivity=4)

    return _pa(kernels[:-1], emb, label, cc, kernel_num, label_num, min_area)
