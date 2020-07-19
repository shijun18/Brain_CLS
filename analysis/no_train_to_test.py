import sys
sys.path.append("..")
from analysis.calculate_sim import calculate_sim_cv,read_and_resize,calculate_ssim
import os
import glob
from tqdm import tqdm
import pandas as pd

def no_train_inf(ad_train_path,cn_train_path,test_path,save_path):
    test_result = {}
    train_path = []
    test_path = glob.glob(os.path.join(test_path,"*.png"))
    test_path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    ad_path =  glob.glob(os.path.join(ad_train_path,"*.png"))
    ad_path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    cn_path =  glob.glob(os.path.join(cn_train_path,"*.png"))
    cn_path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    train_path.extend(ad_path)
    train_path.extend(cn_path)
    # print(train_path)
    print(len(train_path))
    total_label = []
    for test_item in tqdm(test_path):
        test_img = read_and_resize(test_item,size=(128,128))
        result = []
        for train_item in train_path:
            train_img = read_and_resize(train_item,size=(128,128))
            sim = calculate_ssim(test_img,train_img,[-135,-90,-45,0,45,90,135,180])
            result.append(sim)
        if result.index(max(result)) < 1000 :
            label = 'AD'
        else:
            label = 'CN'
        total_label.append(label)
    
    test_result['uuid'] = [ i+1 for i in range(len(test_path))]
    test_result['label'] = total_label
    csv_file = pd.DataFrame(data=test_result)
    csv_file.to_csv(save_path,index=False)


if __name__ == "__main__":

    ad_path = '../dataset/pre_crop_data/train/AD'
    cn_path = '../dataset/pre_crop_data/train/CN'
    test_path = '../dataset/pre_crop_data/test/AD&CN'
    save_path = './no_train_result_ssim.csv'
    no_train_inf(ad_path,cn_path,test_path,save_path)