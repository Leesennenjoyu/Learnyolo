"""
Creat YOLOv1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import resnet18
from utils import SPP, SAM, BottleneckCSP, Conv


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
        self.backbone = resnet(pretrained = True)

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
        grid_y, grid_x = torch.meshgrid([torch.arrange(hs), torch.arrange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim = -1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2).to(self.device)

        return grid_xy



