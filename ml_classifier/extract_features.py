import os
import pandas as pd 



def extract_features(input_csv,save_csv):
  df = pd.read_csv(input_csv)




if __name__ == "__main__":

  input_csv = '../converter/shuffle_crop_label.csv'
  save_csv = '../converter/shuffle_crop_label_features.csv'
  
  extract_features(input_csv,save_csv)