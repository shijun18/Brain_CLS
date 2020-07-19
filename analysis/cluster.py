import pandas as pd
import numpy as np


def read_csv(filename):
    df = pd.read_csv(filename)
    del df['id']
    return df.values


def findfa(id,fa):
    if fa[id] == -1:
        return fa,id
    fa,fa[id] = findfa(fa[id],fa)
    return fa,fa[id]


def merge(id1, id2, fa):
    fa,fa1 = findfa(id1,fa)
    fa,fa2 = findfa(id2,fa)
    if fa1 != fa2:
        fa[fa1] = fa2
    return fa

def union_find(data,threshold=0.7):
    col,row = data.shape
    fa = [-1 for i in range(row)]

    for i in range(row):
        for j in range(col):
            if data[i][j] > threshold:
                fa = merge(i, j, fa)

    rec = []
    result = []
    class_num = 1
    for i in range(row):
        if i % 20 == 0:
            # print("")
            # print('******************%d******************'%class_num)
            class_num += 1
        if findfa(i,fa)[1] == i:
            # print(i + 1, end = ", ")
            rec.append(i+1)
            result.append(i+1)
        else:
            # print(fa[i] + 1, end = ", ")
            result.append(fa[i] + 1)

    print("\ntypes=",end='')
    print(len(rec))
    return result


def merge_class(class_result):
    class_set = list(set(class_result))
    # print(class_set)
    cluster_dict = {}
    for i in range(len(class_set)):
        cluster_dict[str(i+1)] = []
    for index,item in enumerate(class_result):
        key = class_set.index(item)
        cluster_dict[str(key+1)].append(index+1)
    
    # for key in cluster_dict.keys():
    #     print(cluster_dict[key])       
    
    return cluster_dict


if __name__ == "__main__":

    data = read_csv("./sim_csv/AD_merge_cv.csv")
    result = union_find(data,threshold=0.62)
    cluster_dict = merge_class(result) 
    # print(len(result))
    # print(len(set(result)))