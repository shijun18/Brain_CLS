import os
import pandas as pd 
import pandas as pd
import cv2
from skimage.feature import greycomatrix,greycoprops
from skimage.feature import hog
import numpy as np


def get_greymatrix_feature(images,distance,angle,levels,features):
    feature_dict={'contrast':[],'dissimilarity':[],'homogeneity':[],'ASM':[],'energy':[],'correlation':[]}
    for image in images:
        # get greycomatrix
        P = greycomatrix(image,distance,angle,levels)
        # get feature of every comatrix
        for feature in features:
            feature_dict[feature].append(greycoprops(P,feature).ravel())
    return feature_dict




def get_hog_feature(images):
    fog_feature = []
    for image in images:
        fd = hog(image, orientations=4, pixels_per_cell=(32, 32),
                cells_per_block=(1, 1),feature_vector=True)
        fog_feature.append(fd)          

    return fog_feature




def extract_features(input_csv,save_csv):
    df = pd.read_csv(input_csv)

    grey_features = ['contrast','dissimilarity','homogeneity','correlation']
    # grey_features = ['ASM','energy']
    df = pd.read_csv(input_csv)
    id_list = df['id'].values.tolist()
    images_list = []
    for ID in id_list:
        img = cv2.imread(ID,cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
        images_list.append(img_resize)
    grey_feature_dict = get_greymatrix_feature(images_list,[1,2,3,4],[0,np.pi/4,np.pi/2,3*np.pi/4],256,grey_features)
    
    for feature_item in grey_features:
        for i in range(len(grey_feature_dict[feature_item][0])):
            feature_name = feature_item + '_' + str(i)
            df.insert(0,feature_name,list(np.array(grey_feature_dict[feature_item])[:,i]))
    
    # hog_feature = get_hog_feature(images_list)
    # for i in range(len(hog_feature[0])):
    #     feature_name = 'hog_' + str(i)
    #     df.insert(0,feature_name,list(np.array(hog_feature)[:,i]))
    print("Extraction finished!")
    df.to_csv(save_csv,index=False)



if __name__ == "__main__":

  input_csv = '../converter/post_shuffle_crop_label.csv'
  save_csv = '../converter/post_shuffle_crop_label_features.csv'
  
  extract_features(input_csv,save_csv)
