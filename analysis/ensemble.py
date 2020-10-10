import pandas as pd 
import numpy as np  

def vote_ensemble(csv_path_list,save_path,col='label'):
  result = {}
  ensemble_list = []
  for csv_path in csv_path_list:
    csv_file = pd.read_csv(csv_path)
    ensemble_list.append(csv_file[col].values.tolist())

  result['uuid'] = csv_file['uuid'].values.tolist()
  vote_array = np.array(ensemble_list)
  result['label'] = [max(list(vote_array[:,i]),key=list(vote_array[:,i]).count) for i in range(vote_array.shape[1])]
  
  final_csv = pd.DataFrame(result)
  final_csv.to_csv(save_path,index=False)


if __name__ == "__main__":
  
  save_path = './ensemble_csv/new_random_a29_ensem_submission.csv'
  # csv_path_list = ['./ensemble_csv/v1.0_new_submission_ave.csv','./ensemble_csv/v2.0_new_submission_ave.csv','./ensemble_csv/v9.0_new_submission_ave.csv', './ensemble_csv/v7.0_new_submission_ave.csv', \
  #                 './ensemble_csv/v1.0_new_submission_vote.csv', './ensemble_csv/v2.0_new_submission_vote.csv','./ensemble_csv/v9.0_new_submission_vote.csv', './ensemble_csv/v7.0_new_submission_vote.csv', \
  #                  './ensemble_csv/v6.0_new_submission_ave.csv','./ensemble_csv/v6.0_new_submission_vote.csv','./ensemble_csv/v24_0.9227.csv',]
  csv_path_list = ['./ensemble_csv/v2.0_submission_ave.csv', './ensemble_csv/v9.0_new_submission_vote.csv', './ensemble_csv/v24_0.9227.csv']
  reorder = {
    'index_list':[],
    'diff':[]
  }
  from itertools import combinations
  for r in range(2,len(csv_path_list)):
    for index in combinations(range(len(csv_path_list)),r):
      print(index)
      reorder['index_list'].append(index)
      tmp_csv_path_list = [csv_path_list[i] for i in index]
      vote_ensemble(tmp_csv_path_list,save_path)
      from post_process import diff_csv
      diff = diff_csv(save_path,'./ensemble_csv/post_v24_submission_0.9227_0.695.csv','label')
      reorder['diff'].append(diff)

  
  # final_csv = pd.DataFrame(reorder)
  # final_csv.to_csv('./new_diff.csv',index=False)

  # vote_ensemble(csv_path_list,save_path)
  # from post_process import diff_csv
  # diff = diff_csv(save_path,'./ensemble_csv/post_v24_submission_0.9227_0.695.csv','label')