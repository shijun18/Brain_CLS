 
__all__ = ['resnet18','se_resnet18']


NET_NAME = 'se_resnet18'
VERSION = 'v2.0'
DEVICE = '4'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/epoch:95-train_loss:0.04389-val_loss:0.07643.pth'.format(VERSION),
  'se_resnet18':'./ckpt/{}/epoch:78-train_loss:0.04824-val_loss:0.06200.pth'.format(VERSION),
}

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':100,
  'channels':1,
  'num_classes':2,
  'input_shape':(128,128),
  'crop':0,
  'batch_size':16,
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

