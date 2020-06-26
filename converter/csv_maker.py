import os 
import pandas as pd 
import glob
import random

RULE = {"AD":0,
        "CN":1,
        "MCI":2
        }



def make_label_csv(input_path,csv_path):
  '''
  Make label csv file.
  label rule: AD->0, CN->1, MCI->2
  '''
  info = []
  for subdir in os.scandir(input_path):
    # print(subdir.name)
    index = RULE[subdir.name]
    path_list = glob.glob(os.path.join(subdir.path,"*.png"))
    sub_info = [[item,index] for item in path_list]
    info.extend(sub_info)
  
  random.shuffle(info)
  # print(len(info))
  col = ['id','label']
  info_data = pd.DataFrame(columns=col,data=info)
  info_data.to_csv(csv_path,index=False)





if __name__ == "__main__":
  
  input_path = '/staff/shijun/torch_projects/Brain_CLS/dataset/pre_data/train'
  csv_path = './shuffle_label.csv'

  make_label_csv(input_path,csv_path)