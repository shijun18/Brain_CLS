from __future__ import print_function
import numpy as np
from trainer import AverageMeter, accuracy
from run import get_cross_validation, csv_reader_single
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.transforms as tr

import os
import argparse
import logging

import nni

from data_utils.transforms import RandomRotate
from data_utils.data_loader import DataGenerator


_logger = logging.getLogger("brainCLS_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
lr_scheduler = None
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

MEAN = 0.105393
STD = 0.203002
channels = 1
num_classes = 2
FOLD_NUM = 9

input_shape = (128, 128)

train_path = []
val_path = []


def val_on_epoch(epoch):

    net.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for step, sample in enumerate(testloader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            output = net(data)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            acc = accuracy(output.data, target)[0]
            val_loss.update(loss.item(), data.size(0))
            val_acc.update(acc.item(), data.size(0))
            torch.cuda.empty_cache()

    return val_loss.avg, val_acc.avg


def train_on_epoch(epoch):

    net.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for step, sample in enumerate(trainloader):

        data = sample['image']
        target = sample['label']

        data = data.cuda()
        target = target.cuda()

        output = net(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(acc.item(), data.size(0))

        torch.cuda.empty_cache()

        print('Train epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
              .format(epoch, step, loss.item(), acc.item(), optimizer.param_groups[0]['lr']))

    return train_loss.avg, train_acc.avg


def get_net(net_name):
    if net_name == 'resnet18':
        from model.resnet import resnet18
        net = resnet18(input_channels=channels,
                       num_classes=num_classes)
    elif net_name == 'se_resnet18':
        from model.se_resnet import se_resnet18
        net = se_resnet18(input_channels=channels,
                          num_classes=num_classes)
    elif net_name == 'se_resnet10':
        from model.se_resnet import se_resnet10
        net = se_resnet10(input_channels=channels,
                          num_classes=num_classes)
    elif net_name == 'simple_net':
        from model.simple_net import simple_net
        net = simple_net(input_channels=channels,
                         num_classes=num_classes)
    elif net_name == 'tiny_net':
        from model.simple_net import tiny_net
        net = tiny_net(input_channels=channels,
                       num_classes=num_classes)

    return net


def prepare(args, train_path, val_path, label_dict):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global lr_scheduler

    # Data
    print('==> Preparing data..')
    train_transformer = transforms.Compose([
        tr.Resize(input_shape),
        RandomRotate([-135, -90, -45, 0, 45, 90, 135]),
        tr.RandomResizedCrop(input_shape, scale=(0.9, 1.1)),
        tr.RandomHorizontalFlip(p=0.5),
        tr.RandomVerticalFlip(p=0.5),
        tr.ToTensor(),
        tr.Normalize((MEAN,), (STD,))
    ])

    val_transformer = transforms.Compose([
        tr.Resize(size=input_shape),
        tr.ToTensor(),
        tr.Normalize((MEAN,), (STD,))
    ])

    train_dataset = DataGenerator(
        train_path, label_dict, transform=train_transformer)

    trainloader = DataLoader(
        train_dataset,
        batch_size=args['train_batch_size'],
        shuffle=True,
        num_workers=2
    )

    val_dataset = DataGenerator(
        val_path, label_dict, transform=val_transformer)

    testloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    net = get_net(args["net_name"])
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    if args['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(
            net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    elif args['lr_scheduler'] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args['milestones'], gamma=args['gamma'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--cur_fold", type=int, default=0)

    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)

        csv_path = './converter/shuffle_label.csv'
        label_dict = csv_reader_single(
            csv_path, key_col='id', value_col='label')
        path_list = list(label_dict.keys())

        fold_losses = []

        for cur_fold in range(1, FOLD_NUM+1):
            train_path, val_path = get_cross_validation(
                path_list, FOLD_NUM, cur_fold)
            prepare(RCV_CONFIG, train_path, val_path, label_dict)

            fold_best_val_loss = 100.
            for epoch in range(start_epoch, start_epoch+args.epochs):
                epoch_train_loss, epoch_train_acc = train_on_epoch(epoch)
                epoch_val_loss, epoch_val_acc = val_on_epoch(epoch)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                print('Fold %d | Epoch %d | Val Loss %.5f | Acc %.5f'
                      % (cur_fold, epoch, epoch_val_loss, epoch_val_acc))

                fold_best_val_loss = min(fold_best_val_loss, epoch_val_loss)
                nni.report_intermediate_result(epoch_val_loss)

            fold_losses.append(fold_best_val_loss)
            break
        nni.report_final_result(np.mean(fold_losses))
    except Exception as exception:
        _logger.exception(exception)
        raise
