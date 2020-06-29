import os
import argparse
from trainer import Pet_Classifier
import pandas as pd
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER,SETUP_TRAINER,VERSION,CURRENT_FOLD
from sklearn.metrics import classification_report
import time
import random


def get_cross_validation(path_list,fold_num,current_fold):
  
  _len_ = len(path_list) // fold_num
  train_id = []
  validation_id = []
  end_index = current_fold * _len_
  start_index = end_index - _len_
  if current_fold == fold_num:
    validation_id.extend(path_list[start_index:])
    train_id.extend(path_list[:start_index])
  else:
    validation_id.extend(path_list[start_index:end_index])
    train_id.extend(path_list[:start_index])
    train_id.extend(path_list[end_index:])
  
  print(len(train_id),len(validation_id))
  return train_id,validation_id 


def get_parameter_number(net):
  total_num = sum(p.numel() for p in net.parameters())
  trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
  return {'Total': total_num, 'Trainable': trainable_num}



if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', default='train',choices=["train","test","inf"], 
                      help='choose the mode',type=str)
  parser.add_argument('-s', '--save', default='no',choices=['no','n','yes','y'], 
                      help='save the forward middle features or not',type=str)
  parser.add_argument('-p', '--path', default=None, 
                      help='the directory path of input image',type=str)                    
  args = parser.parse_args()
  
  # Set data path & classifier
  csv_path = './converter/shuffle_label.csv'
  label_dict = csv_reader_single(csv_path,key_col='id',value_col='label')
  
  classifier = Pet_Classifier(**INIT_TRAINER)
  print(get_parameter_number(classifier.net))
    
  # Training
  ###############################################
  if args.mode == 'train':
    path_list = list(label_dict.keys())[:1800]
    random.shuffle(path_list)
    for current_fold in range(1,6):
    # train_path,val_path = get_cross_validation(path_list,4,CURRENT_FOLD)
      train_path,val_path = get_cross_validation(path_list,5,current_fold)
      SETUP_TRAINER['train_path']=train_path
      SETUP_TRAINER['val_path']=val_path
      SETUP_TRAINER['label_dict']=label_dict
      start_time = time.time()
      classifier.trainer(**SETUP_TRAINER)
      
      classifier.n_epoch += INIT_TRAINER['n_epoch']
      classifier.start_epoch += INIT_TRAINER['n_epoch']

    print('run time:%.4f'%(time.time()-start_time))
  ###############################################
  
  # Testing
  ###############################################
  elif args.mode == 'test':
    test_path = list(label_dict.keys())[1800:]
    save_path = './analysis/result/{}_test.csv'.format(VERSION)
    
    start_time = time.time()
    if args.save == 'no' or args.save == 'n':
      result,_,_ = classifier.test(test_path,label_dict)
      print('run time:%.4f'%(time.time()-start_time))
    else:
      result,feature_in,feature_out = classifier.test(test_path,label_dict,hook_fn_forward=True)
      print('run time:%.4f'%(time.time()-start_time))
      # save the avgpool output
      print(feature_in.shape,feature_out.shape)
      feature_dir = './analysis/mid_feature/{}'.format(VERSION)
      if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
      from converter.common_utils import save_as_hdf5
      for i in range(len(test_path)):
        name = os.path.basename(test_path[i])
        feature_path = os.path.join(feature_dir,name)
        save_as_hdf5(feature_in[i],feature_path,'feature_in')   
        save_as_hdf5(feature_out[i],feature_path,'feature_out') 
    print(classification_report(result['true'], result['pred'], target_names=['AD','CN'],output_dict=False))
    info = {}
    info['id'] = test_path
    info['label'] = result['true']
    info['pred'] = result['pred']
    info['prob'] = result['prob']
    csv_file = pd.DataFrame(info)
    csv_file.to_csv(save_path,index=False)
  ###############################################

  # Inference
  ###############################################
  elif args.mode == 'inf':
    test_path = [os.path.join(args.path,case) for case in os.listdir(args.path)]
    save_path = './analysis/result/{}_submission.csv'.format(VERSION)
    
    start_time = time.time()
    
    result = classifier.inference(test_path)
    print('run time:%.4f'%(time.time()-start_time))

    info = {}
    info['uuid'] = [os.path.splitext(os.path.basename(case))[0] for case in test_path]
    info['label'] = result['pred']
    info['prob'] = result['prob']
    csv_file = pd.DataFrame(info)
    csv_file.to_csv(save_path,index=False)   
  ###############################################
