from cluster import union_find, read_csv

import pandas as pd
import os


def post_process(input_df, cluster_dict, save_path):
    post_label = input_df['label'].values.tolist()

    for value in cluster_dict.values():
        distrubution = [post_label[index-1] for index in value]
        if distrubution.count('AD') > distrubution.count('CN'):
            label = 'AD'
        else:
            label = 'CN'

        for index in value:
            post_label[index-1] = label

    input_df['post_label'] = post_label
    input_df.to_csv(save_path, index=False)


def merge_class(class_result):
    class_set = list(set(class_result))
    print(class_set)
    cluster_dict = {}
    for i in range(len(class_set)):
        cluster_dict[str(i+1)] = []
    for index, item in enumerate(class_result):
        key = class_set.index(item)
        cluster_dict[str(key+1)].append(index+1)

    for key in cluster_dict.keys():
        print(cluster_dict[key])

    return cluster_dict


if __name__ == "__main__":

    data = read_csv("./test_merge.csv")
    class_result = union_find(data, threshold=0.6)
    print(len(class_result))
    # print(class_result)
    print(len(set(class_result)))

    input_path = './v4-1_submission_0.77262.csv'
    input_df = pd.read_csv(input_path)
    cluster_dict = merge_class(class_result)
    save_path = './post_{}'.format(os.path.basename(input_path))

    post_process(input_df, cluster_dict, save_path)
