 
__all__ = ['resnet18','se_resnet18','se_resnet10','simple_net','tiny_net']


NET_NAME = 'simple_net'
VERSION = 'v4.1'
DEVICE = '1'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/epoch:95-train_loss:0.04389-val_loss:0.07643.pth'.format(VERSION),
  'se_resnet18':'./ckpt/{}/epoch:78-train_loss:0.04824-val_loss:0.06200.pth'.format(VERSION),
  'se_resnet10':'./ckpt/{}/epoch:99-train_loss:0.05755-val_loss:0.08317.pth'.format(VERSION),
  'simple_net':'./ckpt/{}/epoch:84-train_loss:0.10051-val_loss:0.10532.pth'.format(VERSION),
  'tiny_net':'./ckpt/{}/epoch:72-train_loss:0.10209-val_loss:0.12963.pth'.format(VERSION),
}


V3 = {
  'v3.0':'epoch:99-train_loss:0.05755-val_loss:0.08317.pth',
  'v3.0.1':'epoch:87-train_loss:0.08328-val_loss:0.08603.pth',
  'v3.0.2':'epoch:96-train_loss:0.07486-val_loss:0.11337.pth',
  'v3.0.3':'epoch:94-train_loss:0.07525-val_loss:0.11263.pth',
}

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':20,
  'channels':1,
  'num_classes':2,
  'input_shape':(128,128),
  'crop':0,
  'batch_size':32,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH[NET_NAME]
}

# Arguments when perform the trainer 
SETUP_TRAINER = {
  'output_dir':'./ckpt/{}'.format(VERSION),
  'log_dir':'./log/{}'.format(VERSION),
  'optimizer':'Adam',
  'loss_fun':'Cross_Entropy',
  'class_weight':None,
  'lr_scheduler':None
}

