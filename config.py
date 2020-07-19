 
__all__ = ['resnet18', 'se_resnet18', 'se_resnet10', 'simple_net', 'tiny_net', 'resnet34', 'resnet50']


NET_NAME = 'resnet34'
VERSION = 'v6.0'
DEVICE = '1'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 17

WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/fold1/fold:1 epoch:123-train_loss:0.04582-val_loss:0.32073-train_acc:0.98808-val_acc:0.91447.pth'.format(VERSION),
  'se_resnet18':'./ckpt/{}/fold1/epoch:78-train_loss:0.04824-val_loss:0.06200.pth'.format(VERSION),
  'se_resnet10':'./ckpt/{}/fold1/epoch:78-train_loss:0.09650-val_loss:0.10779.pth'.format(VERSION),
  'simple_net':'./ckpt/{}/fold7/fold:7 epoch:122-train_loss:0.07240-val_loss:0.08959-train_acc:0.98077-val_acc:0.97419.pth'.format(VERSION),
  'tiny_net':'./ckpt/{}/fold5/fold:6 epoch:139-train_loss:0.07489-val_loss:0.09525-train_acc:0.97527-val_acc:0.96774.pth'.format(VERSION),
  'resnet34':'./ckpt/{}/fold2/fold:2 epoch:34-train_loss:0.24320-val_loss:0.30565-train_acc:0.90016-val_acc:0.87500.pth'.format(VERSION),
  'resnet50':'./ckpt/{}/fold4/fold:4 epoch:133-train_loss:0.06670-val_loss:0.31578-train_acc:0.97393-val_acc:0.88760.pth'.format(VERSION),
}


WEIGHT_PATH_LIST = [
  './ckpt/{}/fold1/fold:1 epoch:123-train_loss:0.04582-val_loss:0.32073-train_acc:0.98808-val_acc:0.91447.pth'.format(VERSION),
  # './ckpt/{}/fold2/fold:2 epoch:134-train_loss:0.05817-val_loss:0.14187-train_acc:0.98196-val_acc:0.94378.pth'.format(VERSION),
  # './ckpt/{}/fold3/fold:3 epoch:109-train_loss:0.09563-val_loss:0.16289-train_acc:0.96192-val_acc:0.92369.pth'.format(VERSION),
  # './ckpt/{}/fold4/fold:4 epoch:97-train_loss:0.13098-val_loss:0.17313-train_acc:0.95491-val_acc:0.92771.pth'.format(VERSION),
  # './ckpt/{}/fold5/fold:5 epoch:142-train_loss:0.07010-val_loss:0.12331-train_acc:0.97892-val_acc:0.94422.pth'.format(VERSION),
  # './ckpt/{}/fold6/fold:6 epoch:139-train_loss:0.07489-val_loss:0.09525-train_acc:0.97527-val_acc:0.96774.pth'.format(VERSION),
  # './ckpt/{}/fold7/fold:7 epoch:130-train_loss:0.09209-val_loss:0.11351-train_acc:0.96520-val_acc:0.95484.pth'.format(VERSION),
  # './ckpt/{}/fold8/fold:8 epoch:126-train_loss:0.05963-val_loss:0.12561-train_acc:0.98065-val_acc:0.93210.pth'.format(VERSION),
]



# Arguments when trainer initial
INIT_TRAINER = {
    'net_name': NET_NAME,
    'lr': 1e-3,
    'n_epoch': 150,
    'channels': 3,
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
    'lr_scheduler': None
}
