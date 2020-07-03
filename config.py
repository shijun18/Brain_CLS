 
__all__ = ['resnet18','se_resnet18','se_resnet10','simple_net','tiny_net','se_tiny_net']

NET_NAME = 'tiny_net'
VERSION = 'v1.4'
DEVICE = '6'

NET_NAME = 'simple_net'
VERSION = 'v4.3'
DEVICE = '6'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 9
TTA_TIMES = 15

WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/epoch:95-train_loss:0.04389-val_loss:0.07643.pth'.format(VERSION),
  'se_resnet18':'./ckpt/{}/epoch:78-train_loss:0.04824-val_loss:0.06200.pth'.format(VERSION),
  'se_resnet10':'./ckpt/{}/epoch:78-train_loss:0.09650-val_loss:0.10779.pth'.format(VERSION),
  'simple_net':'./ckpt/{}/epoch:185-train_loss:0.08428-val_loss:0.08931.pth'.format(VERSION),
  'tiny_net':'./ckpt/{}/epoch:94-train_loss:0.15337-val_loss:0.13961.pth'.format(VERSION),
  'se_tiny_net':'./ckpt/{}/'.format(VERSION),
}


WEIGHT_PATH_LIST = {}


WEIGHT_PATH_LIST = {
    './ckpt/v1.4/fold1/fold:1 epoch:171-train_loss:0.05034-val_loss:0.06367-train_acc:0.98031-val_cc:0.96847.pth',
    './ckpt/v1.4/fold2/fold:2 epoch:172-train_loss:0.05673-val_loss:0.05404-train_acc:0.97919-val_cc:0.98649.pth',
    './ckpt/v1.4/fold3/fold:3 epoch:177-train_loss:0.06317-val_loss:0.05245-train_acc:0.97582-val_cc:0.98649.pth',
    './ckpt/v1.4/fold4/fold:4 epoch:186-train_loss:0.04054-val_loss:0.07168-train_acc:0.98425-val_cc:0.95946.pth',
    './ckpt/v1.4/fold5/fold:5 epoch:185-train_loss:0.04912-val_loss:0.07480-train_acc:0.98369-val_cc:0.97748.pth',
    './ckpt/v1.4/fold6/fold:6 epoch:145-train_loss:0.06114-val_loss:0.06208-train_acc:0.97863-val_cc:0.97297.pth',
    './ckpt/v1.4/fold7/fold:7 epoch:163-train_loss:0.03610-val_loss:0.08639-train_acc:0.98875-val_cc:0.95495.pth',
    './ckpt/v1.4/fold8/fold:8 epoch:165-train_loss:0.04623-val_loss:0.07621-train_acc:0.98313-val_cc:0.97297.pth',
    './ckpt/v1.4/fold9/fold:9 epoch:191-train_loss:0.03348-val_loss:0.09733-train_acc:0.98930-val_cc:0.96875.pth',
}

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-4,
    'n_epoch': 300,
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
