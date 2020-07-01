import os 
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler



params_dict = {
  'lasso':{
    'C':[0.001,0.01,0.1,1,10]
  },
  'knn':{
    'n_neighbors':range(1,12)
  },
  'svm':{
    'C':[0.001,0.01,0.1,1,10],
    'gamma':[0.001,0.01,0.1,1,10]
  },
  'decision tree':{
    'max_depth':range(3,15),
  },
  'random forest':{
    'n_estimators':range(10,100,5),
    'criterion':['gini','entropy']
  },
  'extra trees':{
    'n_estimators':range(10,100,5),
    'criterion':['gini','entropy']
  },
  'bagging':{
    'n_estimators':range(10,100,5),
  },
  'mlp':{
    'alpha':[0.001,0.01,0.1,1,10],
    'hidden_layer_sizes':[(5,),(10,),(15,),(20,),(10,10),(7,7,7),(5,5,5,5)],
    'solver':['lbfgs'],
    'activation':['logistic'],
    'learning_rate':['constant','invscaling']
  },
  'xgboost':{
    'n_estimators':range(10,50,5),
    'max_depth':range(2,15,1),
    'learning_rate':np.linspace(0.01,2,20),
    'subsample':np.linspace(0.7,0.9,20),
    'colsample_bytree':np.linspace(0.5,0.98,10),
    'min_child_weight':range(1,9,1)
  }
}


METRICS = {
  'Accuracy':'accuracy',
  'Recall':'recall',
  'Precision':'precision',
  'F1':'f1',
  'AUC':'roc_auc'
  }

class ML_Classifier(object):
  '''
  Machine Learning Classifier for the classification
  Args:
  - clf_name, string, __all__ = ['lasso','knn','svm','decision tree','random forest','extra trees','bagging','mlp','xgboost']
  - params_dict, dict, parameters setting of the specified classifer
  '''
  def __init__(self,clf_name=None,params=None): 
    super(ML_Classifier,self).__init__()  
    self.clf_name = clf_name
    self.params = params
    self.clf = self._get_clf()  
  

  def trainer(self,df,target_key,test_size=0.2,random_state=21,metric=None,k_fold=5):
    params = self.params
    data_x,target_y = self.extract_df(df,target_key,random_state) 
    
    x_train,x_test,y_train,y_test = train_test_split(data_x,target_y,
                                            random_state=random_state,
                                            test_size=test_size,
                                            shuffle=True)
  
    kfold = KFold(n_splits=k_fold)
    grid = GridSearchCV(estimator=self.clf,
                        param_grid=params,
                        cv=kfold,
                        scoring=metric,
                        refit='Accuracy',
                        return_train_score=True)
    grid = grid.fit(x_train,y_train)

    best_score = grid.best_score_
    best_model = grid.best_estimator_
    test_score = best_model.score(x_test,y_test)
    print("Accuracy Evaluation:")
    print("Best score:{}".format(best_score))
    print("Test score:{}".format(test_score))
    
    print('Best parameters:')
    for key in params.keys():
      print('%s:'%key)
      print(best_model.get_params()[key])
    
    if self.clf_name == 'random forest' or self.clf_name == 'extra trees':
      if self.clf_name == 'random forest':
        new_grid = RandomForestClassifier(random_state=0,bootstrap=True)
      elif self.clf_name == 'extra trees':
        new_grid = ExtraTreesClassifier(random_state=0,bootstrap=True)
      new_grid.set_params(**grid.best_params_)

      new_grid = new_grid.fit(x_train,y_train) 
      importances = new_grid.feature_importances_
      feat_labels = df.columns
      # print(feat_labels)
      indices = np.argsort(importances)[::-1]

      # print(indices)
      for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1,30, feat_labels[indices[f]], importances[indices[f]]))

    return best_model

    
  def _get_clf(self):
    if self.clf_name == 'lasso':
      classifer = LogisticRegression(penalty='l2',random_state=0)
    elif self.clf_name == 'knn':
      classifer = KNeighborsClassifier()
    elif self.clf_name == 'svm':
      classifer = SVC(kernel='rbf',random_state=0)
    elif self.clf_name == 'decision tree':
      classifer = DecisionTreeClassifier(random_state=0)
    elif self.clf_name == 'random forest':
      classifer = RandomForestClassifier(random_state=0,bootstrap=True)
    elif self.clf_name == 'extra trees':
      classifer = ExtraTreesClassifier(random_state=0,bootstrap=True)
    elif self.clf_name == 'bagging':
      classifer = BaggingClassifier(random_state=0)
    elif self.clf_name == 'mlp':
      classifer = MLPClassifier(max_iter=2000,warm_start=True,random_state=0)
    elif self.clf_name == 'xgboost':
      classifer = XGBClassifier()

    return classifer  


  def extract_df(self,df,target_key,random_state):
    y = df[target_key]
    del df[target_key]
    x = df
    
    return x.values,y.values



