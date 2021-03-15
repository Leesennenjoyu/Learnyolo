from __future__ import division # 用于 python2.7 的除法

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import  torch.backends.cudnn as cudnn

from data import *
import tools

from utils.augmentations import SSDAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='Gamma update for SGD')

    return parser.parse_args()

def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_floder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok = True)

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # muti-scale, 多尺度
    if args.muti_scale:
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]

    cfg = train_cfg

    # dataset and evaluator
    print('Setting Arguments...:', args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')

    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        dataset  = VOCDetection(root = data_dir,
                                img_size = train_size[0],
                                trainform = SSDAugmentation(train_size)
                                )
        evaluator = VOCAPIEvaluator(data_root = data_dir,
                                    img_size = val_size,
                                    device = device,
                                    trainsform = BaseTransform(val_size),
                                    labelmap = VOC_CLASSES
                                    )
    elif args.dataset == 'coco'
        data_dir = coco_root
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size[0],
                    transform=SSDAugmentation(train_size),
                    debug=args.debug
                    )

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size)
                        )

    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = True
        collate_fn = detection_collate,
        num_workers = args.num_workers,
        pin_memory = True
    )

    # build model
    if args.version == 'yolo':
        from model.yolo import myYOLO
        yolo_net = myYOLO(devic , input_size = train_size, num_classes = num_classes, trainable = True)
        print('Let us train yolo on the %s dataset ......' % (args.dataset))
    else:
        print('We only support Yolo !!!')

    model = yolo_net
    model.to(device).train()

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device)) # map_location切换CPU和GPU

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum = args.momentum,
        weight_decay = args.weight_decay
    )

    max_epoch = cfg['max_epoch']
