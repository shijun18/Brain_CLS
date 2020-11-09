import os
import shutil
import glob

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def remove_dir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)



def get_weight_list(ckpt_path):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    
    return path_list



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = glob.glob(os.path.join(ckpt_path,'*.pth'))
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return pth_list[-1]
        else:
            return None
    else:
        return None