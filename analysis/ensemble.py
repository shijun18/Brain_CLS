import pandas as pd 
import numpy as np  

def vote_ensemble(csv_path_list,save_path,col='label'):
  result = {}
  ensemble_list = []
  for csv_path in csv_path_list:
    csv_file = pd.read_csv(csv_path)
    ensemble_list.append(csv_file[col].values.tolist())

  result['uuid'] = csv_file['uuid'].values.tolist()
  ensemble_array = np.array(ensemble_list)
  final_label = list((np.sum(ensemble_array,axis=0)>(len(csv_path_list)//2)).astype(np.uint8))
  result['label'] = ['AD' if case == 0 else 'CN' for case in final_label]
  
  final_csv = pd.DataFrame(result)
  final_csv.to_csv(save_path,index=False)


if __name__ == "__main__":
  
  save_path = './submission.csv'
  csv_path_list = ['./result/v2.2_submission.csv','./result/v3.0_submission.csv','./result/v3.0.1_submission.csv',
  './result/v3.0.2_submission.csv','./result/v3.0.3_submission.csv']

  vote_ensemble(csv_path_list,save_path)