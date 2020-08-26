 
__all__ = ['resnet18', 'se_resnet18', 'se_resnet10', 'simple_net', 'tiny_net', 'resnet34', 'resnet50','densenet121','vgg16']


NET_NAME = 'se_resnet18'
VERSION = 'v2.0'
DEVICE = '4'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 9
TTA_TIMES = 9

WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/'.format(VERSION),
  'se_resnet18':'./ckpt/{}/'.format(VERSION),
  'se_resnet10':'./ckpt/{}/'.format(VERSION),
  'simple_net':'./ckpt/{}/'.format(VERSION),
  'tiny_net':'./ckpt/{}/'.format(VERSION),
  'resnet34':'./ckpt/{}/fold1/fold:1 epoch:141-train_loss:0.03354-val_loss:0.12624-train_acc:0.98865-val_acc:0.96417.pth'.format(VERSION),
  'resnet50':'./ckpt/{}/'.format(VERSION),
  'densenet121':'./ckpt/{}/'.format(VERSION),
  'vgg16':'./ckpt/{}/'.format(VERSION),
}

from utils import get_weight_path,make_dir
make_dir('./ckpt/{}/'.format(VERSION))
WEIGHT_PATH_LIST = get_weight_path('./ckpt/{}/'.format(VERSION))


# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3,
    'n_epoch': 150,
    'channels': 3,
    'num_classes': 3,
    'input_shape': (128, 128),
    'crop': 0,
    'batch_size': 64,
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH[NET_NAME],
    'weight_decay': 0,
    'momentum': 0.9,
    'mean': (0.2223,0.2223,0.2223),
    'std': (0.2447,0.2447,0.2447),
    'gamma': 0.1,
    'milestones': [110]
}

# no_crop     
# 'mean': (0.1055, 0.1055, 0.1055),
# 'std': (0.2028, 0.2028, 0.2028),

#crop
#'mean': (0.2223, 0.2223, 0.2223)
#'std':(0.2447, 0.2447, 0.2447)


# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './ckpt/{}'.format(VERSION),
    'log_dir': './log/{}'.format(VERSION),
    'optimizer': 'Adam',
    'loss_fun': 'Cross_Entropy',
    'class_weight': None,
    'lr_scheduler': 'CosineAnnealingLR'
}
