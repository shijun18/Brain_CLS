import sys
sys.path.append("..")
from analysis.cluster import union_find,read_csv,merge_class

import pandas as pd 
import os 


def post_process(input_df,cluster_dict,save_path):
    post_label = input_df['label'].values.tolist()
    
    for value in cluster_dict.values():
        distrubution = [post_label[index-1] for index in value]
        label = max(distrubution,key=distrubution.count)
        for index in value:
            post_label[index-1] = label  

     
    input_df['post_label'] = post_label
    input_df.to_csv(save_path,index=False)


def diff_csv(pred_csv,target_csv,pred_key='post_label',target_key='label'):
    pred_list = pd.read_csv(pred_csv)[pred_key].values.tolist() 
    target_list = pd.read_csv(target_csv)[target_key].values.tolist() 
    count = 0
    for i,j in zip(pred_list,target_list):
        if i!=j:
            count += 1
    print('diff with target = %d'%count)        
    
    return count


if __name__ == "__main__":

    threshold = 0.67
    threshold_list = []
    for i in range(300):
        print("*****%.3f*****"%threshold)
        data = read_csv("./sim_csv/test_merge.csv")
        class_result = union_find(data,threshold=threshold)
        print("types = %d"%len(set(class_result)))

        input_path = './ensemble_csv/v24_0.9227.csv'
        input_df = pd.read_csv(input_path)
        cluster_dict = merge_class(class_result) 
        save_path = './ensemble_csv/post_{}'.format(os.path.basename(input_path))   

        post_process(input_df,cluster_dict,save_path)

        post_label = pd.read_csv(save_path)['post_label'].values.tolist() 

        AD_num,CN_num,MCI_num = post_label.count('AD'),post_label.count('CN'),post_label.count('MCI')
        max_num = max(abs(AD_num-600),abs(CN_num-600),abs(MCI_num-800))
        print("max diff = %d"%max_num)
        if max_num <= 20:
            print(AD_num,CN_num,MCI_num)
            threshold_list.append(threshold)
            target_csv = './ensemble_csv/post_v24_submission_0.9227_0.695.csv'
            diff = diff_csv(save_path,target_csv)
        
        threshold += 0.001
        
    print(threshold_list)