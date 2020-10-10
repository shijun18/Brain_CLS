import sys
sys.path.append("..")
from analysis.cluster import union_find,read_csv,merge_class

import pandas as pd 
import os 
import random


def class_split_single(csv_path,base_path,threshold=0.8):
    data = read_csv(csv_path)
    result = union_find(data,threshold=threshold)
    class_result = set(result)
    print(len(class_result))
    out_path = []
    for item in class_result:
        case_path = os.path.join(base_path,str(item) + '.png')
        out_path.append(case_path)
    
    return out_path


def get_class_split_single(threshold=0.7):
    
    csv_info = []
    
    save_path = '../converter/pre_{:.2f}_shuffle_crop_label.csv'.format(threshold)

    ad_base = '/staff/shijun/torch_projects/Brain_CLS/dataset/pre_crop_data/train/AD/' 
    ad_csv = "/staff/shijun/torch_projects/Brain_CLS/analysis/sim_csv/AD_merge.csv"
    ad_path = class_split_single(ad_csv,ad_base,threshold=0.8)
    print(len(ad_path))
    csv_info.extend([[case,0] for case in ad_path])
    
    
    cn_base = '/staff/shijun/torch_projects/Brain_CLS/dataset/pre_crop_data/train/CN/' 
    cn_csv = "/staff/shijun/torch_projects/Brain_CLS/analysis/sim_csv/CN_merge.csv"
    cn_path = class_split_single(cn_csv,cn_base,threshold=0.8)
    print(len(cn_path))
    csv_info.extend([[case,1] for case in cn_path])


    random.shuffle(csv_info)
    print(csv_info)
    print(len(csv_info))
    col = ['id','label']
    info_data = pd.DataFrame(columns=col,data=csv_info)
    info_data.to_csv(save_path,index=False)
    


def class_split_multiply(csv_path,base_path,threshold=0.7):
    data = read_csv(csv_path)
    result = union_find(data,threshold=threshold)
    cluster_dict = merge_class(result)
    # print(cluster_dict)
    path_list = []
    for key in cluster_dict.keys():
        item = cluster_dict[key]
        # if len(item) > 20:
        #     random.shuffle(item)
        #     item = item[:20]
        path_list.append([os.path.join(base_path,str(case)+ '.png') for case in item])
    return path_list
    

def get_class_split_multiply(threshold=0.75):
    data_path = []
    ad_base = '/staff/shijun/torch_projects/Brain_CLS/dataset/post_data/train/AD/' 
    ad_csv = "/staff/shijun/torch_projects/Brain_CLS/analysis/sim_csv/AD_merge.csv"
    ad_path = class_split_multiply(ad_csv,ad_base,threshold=threshold)
    data_path.extend(ad_path)
    # print(ad_path)
    print(len(ad_path))

    cn_base = '/staff/shijun/torch_projects/Brain_CLS/dataset/post_data/train/CN/' 
    cn_csv = "/staff/shijun/torch_projects/Brain_CLS/analysis/sim_csv/CN_merge.csv"
    cn_path = class_split_multiply(cn_csv,cn_base,threshold=threshold)
    data_path.extend(cn_path)
    # print(cn_path)
    print(len(cn_path))

    mci_base = '/staff/shijun/torch_projects/Brain_CLS/dataset/post_data/train/MCI/' 
    mci_csv = "/staff/shijun/torch_projects/Brain_CLS/analysis/sim_csv/MCI_merge.csv"
    mci_path = class_split_multiply(mci_csv,mci_base,threshold=threshold)
    data_path.extend(mci_path)
    # print(mci_path)
    print(len(mci_path))
    random.shuffle(data_path)

    return data_path


def get_cross_val_by_class(path_list,fold_num,current_fold):
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
    
    train_path = []
    for case in train_id:
        train_path.extend(case)
    val_path = []
    for case in validation_id:
        val_path.extend(case)

    print("Train set length ", len(train_path),
          "Val set length", len(val_path))
    return train_path, val_path





if __name__ == "__main__":
    data_path = get_class_split_multiply(threshold=0.75)
    # print(data_path)
    print(len(data_path))
    train_path,val_path = get_cross_val_by_class(data_path,9,1)
    

    
    