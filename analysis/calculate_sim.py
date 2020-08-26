import os
import pandas as pd
import cv2
import numpy as np
import glob
from scipy import linalg
from skimage.measure import compare_ssim
from PIL import Image
from tqdm import tqdm

def read_and_resize(img_path,size=(128,128)):

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img_resize = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)

    return img_resize     


def calculate_fid(act1, act2, eps=1e-6):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)




def calculate_hist_sim(img_1,img_2):
    H1 = cv2.calcHist([img_1],[0],None,[256],[0,255])
    H2 = cv2.calcHist([img_2],[0],None,[256],[0,255])
    sim = cv2.compareHist(H1, H2, 4)
    return sim



def similarity_cal_hist(input_path,save_path,resize_flag=False):
    path_list = glob.glob(os.path.join(input_path,"*.*g"))
    path_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    path_list = path_list[:22]
    
    csv_data = {}
    sim_matrix = np.zeros((len(path_list),len(path_list)),dtype=np.float32)
    for i in tqdm(range(len(path_list))):
        for j in range(i,len(path_list)):
            if resize_flag:
                img_1 = read_and_resize(path_list[i],size=(128,128))
                img_2 = read_and_resize(path_list[j],size=(128,128))
            else:    
                img_1 = np.asarray(Image.open(path_list[i]).convert('L')) 
                img_2 = np.asarray(Image.open(path_list[j]).convert('L')) 
            
            if img_1.shape == img_2.shape:
                sim = calculate_hist_sim(img_1,img_2)
            else:
                sim = 0.0    
            sim_matrix[i][j] = sim
            # print(sim)
    index_low = np.tril_indices(len(path_list))
    sim_matrix[index_low] = np.tril(sim_matrix.T)[index_low]

    print('sim_matrix done!')
    csv_data['id'] = [ i+1 for i in range(len(path_list))]
    csv_file = pd.DataFrame(data=csv_data)
    for i in reversed(range(len(path_list))):
        csv_file.insert(1,str(i+1),list(sim_matrix[:,i]))
    
    csv_file.to_csv(save_path,index=False)





def calculate_sim_cv(img_1,img_2,mode='surf'):
    if mode == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        # sift = cv2.SIFT()
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)
    elif mode == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        # surf = cv2.SURF()
        kp1, des1 = surf.detectAndCompute(img_1,None)
        kp2, des2 = surf.detectAndCompute(img_2,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    sim = len(good) / len(kp2)
    return sim




def similarity_cal_cv(input_path,save_path,resize_flag=False,mode='surf'):
    path_list = glob.glob(os.path.join(input_path,"*.*g"))
    path_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # path_list = path_list[:22]
    
    csv_data = {}
    sim_matrix = np.zeros((len(path_list),len(path_list)),dtype=np.float32)
    for i in tqdm(range(len(path_list))):
        for j in range(i,len(path_list)):
            if resize_flag:
                img_1 = read_and_resize(path_list[i],size=(128,128))
                img_2 = read_and_resize(path_list[j],size=(128,128))
            else:    
                img_1 = np.asarray(Image.open(path_list[i]).convert('L')) 
                img_2 = np.asarray(Image.open(path_list[j]).convert('L')) 
            
            if img_1.shape == img_2.shape:
                sim = calculate_sim_cv(img_1,img_2,mode)
            else:
                sim = 0.0    
            sim_matrix[i][j] = sim
            # print(sim)
    index_low = np.tril_indices(len(path_list))
    sim_matrix[index_low] = np.tril(sim_matrix.T)[index_low]

    print('sim_matrix done!')
    csv_data['id'] = [ i+1 for i in range(len(path_list))]
    csv_file = pd.DataFrame(data=csv_data)
    for i in reversed(range(len(path_list))):
        csv_file.insert(1,str(i+1),list(sim_matrix[:,i]))
    
    csv_file.to_csv(save_path,index=False)






def calculate_ssim(img_1,img_2,rotation=None):
    if rotation is None:
        return compare_ssim(img_1,img_2)
    else:
        ssim_rotation = []
        for angle in rotation:
            tmp_img_2 = Image.fromarray(np.copy(img_2)).convert('L')
            ssim = compare_ssim(img_1,np.asarray(tmp_img_2.rotate(angle)))
            ssim_rotation.append(ssim)

        return max(ssim_rotation)    

   

def similarity_cal(input_path,save_path,resize_flag=False,rotate_flag=False):
    path_list = glob.glob(os.path.join(input_path,"*.*g"))
    path_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    csv_data = {}
    sim_matrix = np.zeros((len(path_list),len(path_list)),dtype=np.float32)
    for i in tqdm(range(len(path_list))):
        for j in range(i,len(path_list)):
            if resize_flag:
                img_1 = read_and_resize(path_list[i],size=(128,128))
                img_2 = read_and_resize(path_list[j],size=(128,128))
            else:    
                img_1 = np.asarray(Image.open(path_list[i]).convert('L')) 
                img_2 = np.asarray(Image.open(path_list[j]).convert('L')) 
            
            if img_1.shape == img_2.shape:
                if rotate_flag:
                    sim = calculate_ssim(img_1,img_2,[-135,-90,-45,0,45,90,135,180])
                else:
                    sim = calculate_ssim(img_1,img_2)
            else:
                sim = 0.0    
            sim_matrix[i][j] = sim
            # print(sim)
    index_low = np.tril_indices(len(path_list))
    sim_matrix[index_low] = np.tril(sim_matrix.T)[index_low]

    print('sim_matrix done!')
    csv_data['id'] = [ i+1 for i in range(len(path_list))]
    csv_file = pd.DataFrame(data=csv_data)
    for i in reversed(range(len(path_list))):
        csv_file.insert(1,str(i+1),list(sim_matrix[:,i]))
    
    csv_file.to_csv(save_path,index=False)



def meger_csv(path_s,path_d,save_path):
    df_s = pd.read_csv(path_s)
    del df_s['id']
    s_array = df_s.values
    
    df_d = pd.read_csv(path_d)
    del df_d['id']
    d_array = df_d.values

    bool_array = (s_array != 0.0).astype(np.float32)
    d_array = np.multiply(d_array,bool_array)
    

    csv_data = {}
    csv_data['id'] = [ i+1 for i in range(d_array.shape[0])]
    csv_file = pd.DataFrame(data=csv_data)
    for i in reversed(range(d_array.shape[0])):
        csv_file.insert(1,str(i+1),list(d_array[:,i]))
    
    csv_file.to_csv(save_path,index=False)


if __name__ == "__main__":
    
    # data_path  = '../dataset/post_data/train/AD'
    # save_csv = './sim_csv/AD_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=False)
    # print('AD done!')
    # data_path  = '../dataset/post_data/train/CN'
    # save_csv = './sim_csv/CN_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=False)
    # print('CN done!')
    # data_path  = '../dataset/post_data/train/MCI'
    # save_csv = './sim_csv/MCI_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=False)
    # print('MCI done!')
    
    # data_path  = '../dataset/post_crop_data/train/AD'
    # save_csv = './sim_csv/AD_crop_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=True)
    # print('AD done!')
    # data_path  = '../dataset/post_crop_data/train/CN'
    # save_csv = './sim_csv/CN_crop_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=True)
    # print('CN done!')
    # data_path  = '../dataset/post_crop_data/train/MCI'
    # save_csv = './sim_csv/MCI_crop_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=True)
    # print('MCI done!')

    # path_sim = './sim_csv/AD_sim.csv'
    # path_crop_sim = './sim_csv/AD_crop_sim.csv'
    # save_path = './sim_csv/AD_merge.csv'
    # meger_csv(path_sim,path_crop_sim,save_path)
    # print('AD done!')

    # path_sim = './sim_csv/CN_sim.csv'
    # path_crop_sim = './sim_csv/CN_crop_sim.csv'
    # save_path = './sim_csv/CN_merge.csv'
    # meger_csv(path_sim,path_crop_sim,save_path)
    # print('CN done!')

    path_sim = './sim_csv/MCI_sim.csv'
    path_crop_sim = './sim_csv/MCI_crop_sim.csv'
    save_path = './sim_csv/MCI_merge.csv'
    meger_csv(path_sim,path_crop_sim,save_path)
    print('MCI done!')
    
    
    # data_path  = '../dataset/post_data/test/AD&CN&MCI'
    # save_csv = './sim_csv/test_sim.csv'
    # similarity_cal(data_path,save_csv)
    # print('sim done!')

    # data_path  = '../dataset/post_crop_data/test/AD&CN&MCI'
    # save_csv = './sim_csv/test_crop_sim.csv'
    # similarity_cal(data_path,save_csv,resize_flag=True,rotate_flag=True)
    # print('crop sim done!')

    # path_sim = './sim_csv/test_sim.csv'
    # path_crop_sim = './sim_csv/test_crop_sim.csv'
    # save_path = './sim_csv/test_merge.csv'
    # meger_csv(path_sim,path_crop_sim,save_path)
    # print('test done!')
