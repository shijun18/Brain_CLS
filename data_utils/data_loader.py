import sys
sys.path.append('..')
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - label_dict: dict, file path as key, label as value
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, label_dict, transform=None):

    self.path_list = path_list
    self.label_dict = label_dict
    self.transform = transform


  def __len__(self):
    return len(self.path_list)


  def __getitem__(self,index):
    # Get image and label
    # image: D,H,W
    # label: integer, 0,1,..
    image = Image.open(self.path_list[index]).convert('L')
    # print(self.path_list[index])
    # assert len(image.shape) == 3
    label = self.label_dict[self.path_list[index]]    
    # Transform
    if self.transform is not None:
      image = self.transform(image)
    sample = {'image':image, 'label':np.uint8(label)}
    return sample