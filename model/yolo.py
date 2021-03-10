"""
Creat YOLOv1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


"""1. 创建Yolo网络"""
class myYOLO(nn.Module):
    
    def __init__(self, device, input_size = None, num_classes = 20, trainable = Falese, conf_thresh = 0.01, nms_thresh = 0.5, hr = False):
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
        self.backbone = res


