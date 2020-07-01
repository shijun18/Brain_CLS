import os
import pandas as pd
import PIL
import cv2
from skimage.feature import greycomatrix,greycoprops
import numpy as np
def read_image(dir,image_type):
    '''
    image_type: cv2.IMREAD_COLOR(no alpha),cv2.IMREAD_GRAYSCALE,cv2.IMREAD_UNCHANGED
    '''
    images=[]
    for pic_path in os.listdir(dir):
        path=os.path.join(dir,pic_path)
        img=cv2.imread(path,image_type)
        img_resize=cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
        images.append(img_resize)
    return np.array(images)

def get_greymatrix_feature(images,distance,angle,levels,features):
    feature_dic={'contrast':[],'dissimilarity':[],'homogeneity':[],'ASM':[],'energy':[],'correlation':[]}
    for image in images:
        # get greycomatrix
        P = greycomatrix(image,distance,angle,levels)
        # get feature of every comatrix
        for feature in features:
            feature_dic[feature].append(greycoprops(P,feature).ravel())
    return feature_dic

def extract_features(input_csv,save_csv):
    features = ['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
    df = pd.read_csv(input_csv)
    id = df['id'].values.tolist()
    images_tmp = []
    for path in id:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
        images_tmp.append(img_resize)
    images = np.array(images_tmp)
    feature_dic = get_greymatrix_feature(images,[1,2,3,4],[0,np.pi/4,np.pi/2,3*np.pi/4],256,features)
    df_write = pd.DataFrame(feature_dic)
    df_write.to_csv(save_csv)

if __name__ == "__main__":
    input_csv = '../converter/shuffle_crop_label.csv'
    save_csv = '../converter/shuffle_crop_label_features.csv'
    extract_features(input_csv,save_csv)