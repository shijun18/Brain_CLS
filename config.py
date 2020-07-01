 
__all__ = ['resnet18','se_resnet18','se_resnet10','simple_net','tiny_net','se_tiny_net']


NET_NAME = 'simple_net'
VERSION = 'v4.3'
DEVICE = '6'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 8
TTA_TIMES = 33

WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/epoch:95-train_loss:0.04389-val_loss:0.07643.pth'.format(VERSION),
  'se_resnet18':'./ckpt/{}/epoch:78-train_loss:0.04824-val_loss:0.06200.pth'.format(VERSION),
  'se_resnet10':'./ckpt/{}/epoch:78-train_loss:0.09650-val_loss:0.10779.pth'.format(VERSION),
  'simple_net':'./ckpt/{}/epoch:185-train_loss:0.08428-val_loss:0.08931.pth'.format(VERSION),
  'tiny_net':'./ckpt/{}/epoch:94-train_loss:0.15337-val_loss:0.13961.pth'.format(VERSION),
  'se_tiny_net':'./ckpt/{}/'.format(VERSION),
}


WEIGHT_PATH_LIST = {}



# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3,
    'n_epoch': 150,
    'channels': 1,
    'num_classes': 2,
    'input_shape': (128, 128),
    'crop': 0,
    'batch_size': 32,
    'num_workers': 2,
    'device': DEVICE,
    'pre_trained': PRE_TRAINED,
    'weight_path': WEIGHT_PATH[NET_NAME],
    'weight_decay': 0,
    'momentum': 0.9,
    'mean': 0.105393,
    'std': 0.203002,
    'gamma': 0.1,
    'milestones': [110]
}

# Arguments when perform the trainer
SETUP_TRAINER = {
    'output_dir': './ckpt/{}'.format(VERSION),
    'log_dir': './log/{}'.format(VERSION),
    'optimizer': 'Adam',
    'loss_fun': 'Cross_Entropy',
    'class_weight': None,
    'lr_scheduler': None
}
