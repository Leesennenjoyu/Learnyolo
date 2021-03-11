"""
Creat YOLOv1
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from backbone import resnet18
from utils import SPP, SAM, BottleneckCSP, Conv


"""1. 创建Yolo网络"""
class myYOLO(nn.Module):
    
    def __init__(self, device, input_size = None, num_classes = 20, trainable = False, conf_thresh = 0.01, nms_thresh = 0.5, hr = False):

        super(myYOLO).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device = device).float()

        # Use Resnet18 as backbone
        self.backbone = resnet18(pretrained = True)

        # neck
        self.SPP = nn.Sequential(
            Conv(512, 256, k = 1),
            SPP(),
            BottleneckCSP(256 * 4, 512, n = 1, shortcut = False)
        )
        self.SAM = SAM(512)
        self.conv_set = BottleneckCSP(512, 512, n = 3, shortcut = False)

        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def create_grid(self, input_size):

        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = w // self.stride, h//self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim = -1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2).to(self.device)

        return grid_xy

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device = self.device).float()

    def decode_boxes(self, pred):
        """
        input box: [tx, ty, tw, th]
        output box: [xmin, ymin, xmax, ymax]
        """

        output = torch.zero_like(pred)
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2

        return output

    def nms(self, det, scores):
        """
        Pure Python NMS baseline
        """
        x1 = det[:, 0] # xmin
        y1 = det[:, 1] # ymin
        x2 = det[:, 2] # xmax
        y2 = det[:, 3] # ymax

        areas = (x2 - x1) *(y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                      # sort bounding boxes by decreasing order

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], x1[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)

            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reverse all the boundingbox whose over less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange = True, im_shape = None):
        """
        bbox_pred: (HxW, 4), bsize = 1
        prob_pred: (HxW, num_classes), bsize = 1
        """

        bbox_pred = all_local; prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis = 1)
        prob_pred = prob_pred[(np.arrange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()

        # threshold
        keep = np.zeros(len(bbox_pred), dtype = np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]



