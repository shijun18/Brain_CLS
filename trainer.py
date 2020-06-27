import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math

from torch.nn import functional as F

import torchvision.transforms as tr
from data_utils.data_loader import DataGenerator

import torch.distributed as dist

# GPU version.

class Pet_Classifier(object):
  '''
  Control the training, evaluation, test and inference process.
  Args:
  - net_name: string, __all__ = [''].
  - lr: float, learning rate.
  - n_epoch: integer, the epoch number
  - channels: integer, the channel number of the input
  - num_classes: integer, the number of class
  - input_shape: tuple of integer, input dim
  - crop: integer, cropping size
  - batch_size: integer
  - num_workers: integer, how many subprocesses to use for data loading.
  - device: string, use the specified device
  - pre_trained: True or False, default False
  - weight_path: weight path of pre-trained model
  '''
  def __init__(self,net_name=None,lr=1e-3,n_epoch=1,channels=1,num_classes=3,input_shape=None,crop=48,
                batch_size=6,num_workers=0,device=None,pre_trained=False,weight_path=None,weight_decay=0.,momentum=0.95): 
    super(Pet_Classifier,self).__init__()    

    self.net_name = net_name
    self.lr = lr
    self.n_epoch = n_epoch
    self.channels = channels
    self.num_classes = num_classes
    self.input_shape = input_shape
    self.crop = crop
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.device = device
    self.net = self._get_net(self.net_name)
    self.pre_trained = pre_trained
    self.weight_path = weight_path
    self.start_epoch = 0
    self.global_step = 0
    self.loss_threshold = 1.0
    # save the middle output
    self.feature_in = []
    self.feature_out = []
    self.weight_decay = weight_decay
    self.momentum = momentum
    
    os.environ['CUDA_VISIBLE_DEVICES'] = self.device
    
    if self.pre_trained:
      self._get_pre_trained(self.weight_path)
      self.loss_threshold = eval(os.path.splitext(self.weight_path.split(':')[-1])[0])

    

  def trainer(self,train_path,val_path,label_dict,output_dir=None,log_dir=None,optimizer='Adam',
                loss_fun='Cross_Entropy',class_weight=None,lr_scheduler=None):
    
    torch.manual_seed(0)
    print('Device:{}'.format(self.device))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    self.writer = SummaryWriter(log_dir)
    self.global_step = self.start_epoch * math.ceil(len(train_path)/self.batch_size)

    net = self.net
    lr = self.lr
    loss = self._get_loss(loss_fun,class_weight) 
    weight_decay = self.weight_decay
    momentum = self.momentum
    
    if len(self.device.split(',')) > 1:
      net = DataParallel(net)

    # dataloader setting
    train_transformer = transforms.Compose([
      tr.Resize(size=self.input_shape),
      tr.RandomRotation(degrees=30),
      tr.RandomHorizontalFlip(p=0.5),
      tr.RandomVerticalFlip(p=0.5),
      tr.ToTensor()
    ])

    train_dataset = DataGenerator(train_path,label_dict,transform=train_transformer)

    train_loader = DataLoader(
      train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=True
    )

    # copy to gpu
    net = net.cuda()
    loss = loss.cuda()

    # optimizer setting
    optimizer = self._get_optimizer(optimizer,net,lr,weight_decay,momentum)
    if self.pre_trained:
      checkpoint = torch.load(self.weight_path)
      optimizer.load_state_dict(checkpoint['optimizer']) 

    if lr_scheduler is not None:
      lr_scheduler = self._get_lr_scheduler(lr_scheduler,optimizer)


    # acc_threshold = 0.5
    for epoch in range(self.start_epoch,self.n_epoch):
      train_loss,train_acc = self._train_on_epoch(epoch,net,loss,optimizer,train_loader)
      
      torch.cuda.empty_cache()
      
      val_loss,val_acc = self._val_on_epoch(epoch,net,loss,val_path,label_dict)
      
      if lr_scheduler is not None:
        lr_scheduler.step(val_loss)
      

      print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'
        .format(epoch,train_loss,val_loss))
      
      print('epoch:{},train_acc:{:.5f},val_acc:{:.5f}'
        .format(epoch,train_acc,val_acc))


      self.writer.add_scalars(
        'data/loss',{'train':train_loss,'val':val_loss},epoch
      )
      self.writer.add_scalars(
        'data/acc',{'train':train_acc,'val':val_acc},epoch
      )
      self.writer.add_scalar(
        'data/lr',optimizer.param_groups[0]['lr'],epoch
      )


      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      
      if val_loss < self.loss_threshold:
        self.loss_threshold = val_loss
        
        if len(self.device.split(',')) > 1:
          state_dict = net.module.state_dict()
        else:
          state_dict = net.state_dict()

        saver = {
          'epoch':epoch,
          'save_dir':output_dir,
          'state_dict':state_dict,
          'optimizer':optimizer.state_dict()
        }  
        
        file_name = 'epoch:{}-train_loss:{:.5f}-val_loss:{:.5f}.pth'.format(epoch,train_loss,val_loss) 
        save_path = os.path.join(output_dir,file_name)

        torch.save(saver,save_path)

        
    self.writer.close()


  def _train_on_epoch(self,epoch,net,criterion,optimizer,train_loader):

    net.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for step,sample in enumerate(train_loader):
      
      data = sample['image']
      target = sample['label']

      data = data.cuda()
      target = target.cuda()

      output = net(data)
      loss = criterion(output,target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      output = output.float()
      loss = loss.float()
      
      # measure accuracy and record loss
      acc = accuracy(output.data,target)[0]
      train_loss.update(loss.item(),data.size(0))
      train_acc.update(acc.item(),data.size(0))

      torch.cuda.empty_cache()
     
      print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'
        .format(epoch,step,loss.item(),acc.item(),optimizer.param_groups[0]['lr']))
      
      if self.global_step%10==0: 
        self.writer.add_scalars(
          'data/train_loss_acc',{'train_loss':loss.item(),'train_acc':acc.item()},self.global_step
        )

      self.global_step += 1  

    return train_loss.avg,train_acc.avg
  

  def _val_on_epoch(self,epoch,net,criterion,val_path,label_dict):

    net.eval()
    
    val_transformer = transforms.Compose([
      tr.Resize(size=self.input_shape),
      tr.ToTensor()
    ])

    val_dataset = DataGenerator(val_path,label_dict,transform=val_transformer)

    val_loader = DataLoader(
      val_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )

    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    with torch.no_grad():
      for step,sample in enumerate(val_loader):
        data = sample['image']
        target = sample['label']

        data = data.cuda()
        target = target.cuda()

        output = net(data)
        loss = criterion(output,target)

        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        acc = accuracy(output.data,target)[0]
        val_loss.update(loss.item(),data.size(0))
        val_acc.update(acc.item(),data.size(0))

        torch.cuda.empty_cache()

        print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'
          .format(epoch,step,loss.item(),acc.item()))
      

    return val_loss.avg,val_acc.avg

  
  def hook_fn_forward(self,module,input,output):
    # print(module)
    # print(input[0].size()) 
    # print(output.size())
    
    for i in range(input[0].size(0)):
      self.feature_in.append(input[0][i].cpu().numpy())
      self.feature_out.append(output[i].cpu().numpy())  
  

  def test(self,test_path,label_dict,net=None,hook_fn_forward=False):
    
    if net is None:
      net = self.net
    
    if hook_fn_forward:
      net.avgpool.register_forward_hook(self.hook_fn_forward)

    net = net.cuda()
    net.eval()
    
    test_transformer = transforms.Compose([
      tr.Resize(size=self.input_shape),
      tr.ToTensor()
    ])

    test_dataset = DataGenerator(test_path,label_dict,transform=test_transformer)

    test_loader = DataLoader(
      test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    
    result = {
      'true':[],
      'pred':[],
      'prob':[]
    }

    test_acc = AverageMeter()

    with torch.no_grad():
      for step,sample in enumerate(test_loader):
        data = sample['image']
        target = sample['label']

        data = data.cuda()
        target = target.cuda() #N

        output = net(data)
        output = output.float() #N*C
        
        acc = accuracy(output.data,target)[0]
        test_acc.update(acc.item(),data.size(0))

        result['true'].extend(target.detach().tolist())
        result['pred'].extend(torch.argmax(output,1).detach().tolist())
        output = F.softmax(output,dim=1)
        result['prob'].extend(output.detach().tolist())
        
        print('step:{},test_acc:{:.5f}'
          .format(step,acc.item()))

        torch.cuda.empty_cache()
    
    print('average test_acc:{:.5f}'.format(test_acc.avg))
 
    
    return result,np.array(self.feature_in),np.array(self.feature_out)
  

  
  def inference(self,test_path,label_dict=None,net=None):
    
    if net is None:
      net = self.net

    net = net.cuda()
    net.eval()
    
    test_transformer = transforms.Compose([
      tr.Resize(size=self.input_shape),
      tr.ToTensor()
    ])

    test_dataset = DataGenerator(test_path,transform=test_transformer)

    test_loader = DataLoader(
      test_dataset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=True
    )
    
    result = {
      'pred':[],
      'prob':[]
    }


    with torch.no_grad():
      for step,sample in enumerate(test_loader):
        data = sample['image']
        data = data.cuda()

        output = net(data)
        output = output.float() #N*C

        result['pred'].extend(torch.argmax(output,1).detach().tolist())
        output = F.softmax(output,dim=1)
        result['prob'].extend(output.detach().tolist())
        
        torch.cuda.empty_cache()
    
    return result



  def _get_net(self,net_name):
    if net_name == 'resnet18':
      from model.resnet import resnet18
      net = resnet18(input_channels=self.channels,num_classes=self.num_classes)
    elif net_name == 'se_resnet18':
      from model.se_resnet import se_resnet18
      net = se_resnet18(input_channels=self.channels,num_classes=self.num_classes)
    return net  


  def _get_loss(self,loss_fun,class_weight=None):
    if class_weight is not None:
      class_weight = torch.tensor(class_weight)

    if loss_fun == 'Cross_Entropy':  
      loss = nn.CrossEntropyLoss(class_weight)  

    return loss  


  def _get_optimizer(self,optimizer,net,lr,weight_decay,momentum):
    if optimizer == 'Adam':
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  

    elif optimizer == 'SGD':
      optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer   


  def _get_lr_scheduler(self,lr_scheduler,optimizer):
    if lr_scheduler == 'ReduceLROnPlateau':
      lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                       mode='min',patience=5,verbose=True)
    
    return lr_scheduler
  
  
  def _get_pre_trained(self,weight_path):
    checkpoint = torch.load(weight_path)
    self.net.load_state_dict(checkpoint['state_dict']) 
    self.start_epoch = checkpoint['epoch'] + 1

    



# computing tools

class AverageMeter(object):
  '''
  Computes and stores the average and current value
  '''
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
  '''
  Computes the precision@k for the specified values of k
  '''
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk,1,True,True)
  pred = pred.t()
  correct = pred.eq(target.view(1,-1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(1/batch_size))
  return res