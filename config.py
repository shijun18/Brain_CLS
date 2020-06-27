 
__all__ = ['resnet18','se_resnet18']


NET_NAME = 'resnet18'
VERSION = 'v1.0'
DEVICE = '0'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


WEIGHT_PATH = {
  'resnet18':'./ckpt/{}/'.format(VERSION),
  'se_resnet18':'./ckpt/{}/'.format(VERSION),
  
}

# Arguments when trainer initial
INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':40,
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

