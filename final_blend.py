
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def process_data(DATA_DIR):
    
    train = pd.read_csv(DATA_DIR+"\\train_s3TEQDk.csv")
    test = pd.read_csv(DATA_DIR+"\\test_mSzZ8RL.csv")
    sub= pd.read_csv(DATA_DIR+"\\sample_submission_eyYijxG.csv")
    
    test_region_list=test['Region_Code'].tolist()
    train=train[train['Region_Code'].isin(test_region_list)]
    
    target=train[['Is_Lead']]
    
    lgbmpred = pd.read_csv(DATA_DIR+'\\lgbmpred.csv')
    xgbpred = pd.read_csv(DATA_DIR+'\\xgbmpred.csv')
    catboostpred = pd.read_csv(DATA_DIR+'\\catboostpred.csv')
    
    total_pred = pd.concat([lgbmpred,xgbpred,catboostpred], axis=1)
    
    lgbmoof = pd.read_csv(DATA_DIR+'\\lgbmoof.csv')
    xgboof = pd.read_csv(DATA_DIR+'\\xgbmoof.csv')
    catboostoof = pd.read_csv(DATA_DIR+'\\catboostoof.csv')
    
    total_oof = pd.concat([lgbmoof,xgboof,catboostoof], axis=1)
    
    
    return train,target,sub,test,total_pred,total_oof

def findbestweight(df1,df2,target):
    max_roc = -1
    max_weight = 0
    max_ensemble_oof  = 0
    weights_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for weight in weights_list:
        ensemble_oof = weight*df1 + (1-weight)*df2
        roc_score = roc_auc_score(target,ensemble_oof)
        if roc_score > max_roc:
            max_ensemble_oof = ensemble_oof
            max_roc = roc_score
            max_weight = weight
    print("The best weights for blending is {0} with AUC {1}".format(max_weight, max_roc))
    return max_weight

def blend():
    train,target,sub,test,total_pred,total_oof=process_data("E:\FILES\VARIOUS PYTHON TASKS\AV LEAD")
    import os
    os.chdir("E:\FILES\VARIOUS PYTHON TASKS\AV LEAD")
    weight1=findbestweight(total_oof['lgbmoof'],total_oof['xgboof'],target)
    lgb_xgb=weight1*total_oof['lgbmoof'] +(1-weight1)*total_oof['xgboof']
    
    weight2=findbestweight(lgb_xgb,total_oof['catboostoof'],target)
    lgb_xgb_cat=weight2*lgb_xgb +(1-weight2)*total_oof['catboostoof']
    
    lgb_xgb_cat_pred=(weight1*total_pred['lgbmpred']+(1-weight1)*total_pred['xgbpred'])*weight2+total_pred['catboostpred']*(1-weight2)
    
    sub['Is_Lead']=lgb_xgb_cat_pred
    #sub['Is_Lead'] = sub.apply(lambda X:1 if X.Is_Lead >= 0.225 else 0, axis=1)
    sub.to_csv('blend_added_na_introduced_hyperopt.csv',index=False)
    print(sub)
    from datetime import datetime

    now = datetime.now()
    
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

blend()

