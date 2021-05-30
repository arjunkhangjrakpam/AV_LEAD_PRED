

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import os


"""### Read data and perform basic preprocessing"""

def process_data(DATA_DIR):
    
    train = pd.read_csv(DATA_DIR+"\\train_s3TEQDk.csv")
    test = pd.read_csv(DATA_DIR+"\\test_mSzZ8RL.csv")
    
    #Removes train rows which has Region_Code not present in test set
    test_region_list=test['Region_Code'].tolist()
    train=train[train['Region_Code'].isin(test_region_list)]
    
    train['train_or_test']='train'
    test['train_or_test']='test'
    df=pd.concat([train,test])
    
    #df['Credit_Product']=(df['Credit_Product'].replace([np.nan],['not_given'])).astype(str)
    
    le = LabelEncoder()
    for col in ['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code','Vintage', 'Credit_Product', 'Is_Active']:
        df[col]=  df[col].astype('str')
        df[col]= le.fit_transform(df[col])
        

    
    return train,test,df

"""### Feature Engineering"""

def frequency_encoding(column_name,output_column_name,df):
    fe_pol = (df.groupby(column_name).size()) / len(df)
    df[output_column_name] = df[column_name].apply(lambda x : fe_pol[x])


def feature_engineering(df):
    le = LabelEncoder()
    
     #Interaction Feature (Combining 2 categorical features and performing frequency encoding)
        
    cat_features=[]
    le_features=[]
    columns=['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code','Vintage', 'Credit_Product',  'Is_Active']

    comb = combinations(columns, 2) 

    for i in list(comb):  
        df[f'{i[0]}_{i[1]}']=df[i[0]].astype(str)+'_'+df[i[1]].astype(str)
        df[f'{i[0]}_{i[1]}_le']=le.fit_transform(df[f'{i[0]}_{i[1]}'])
        le_features.append(f'{i[0]}_{i[1]}_le')
        frequency_encoding(f'{i[0]}_{i[1]}',f'{i[0]}_{i[1]}',df)
        cat_features.append(f'{i[0]}_{i[1]}')   
        
    #Frequency Encoding
    
    frequency_encoding('Region_Code','Region_Code_fe',df)
    #frequency_encoding('Reco_Policy_Cat','Reco_Policy_Cat_fe',df)
    
    #Deriving characteristics of each city by creating aggregate features
    region_aggregate_features = df.groupby(['Region_Code']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Channel_Code': ['nunique','count'],       
                                                    'Gender': ['nunique','count'],
                                                     'Occupation': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    
    region_aggregate_features.columns = ['region_aggregate_features' + '_'.join(c).strip('_') for c in region_aggregate_features.columns]
    df = pd.merge(df, region_aggregate_features, on = ['Region_Code'], how='left')
    
    channel_region_aggregate_features = df.groupby(['Channel_Code','Region_Code']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Gender': ['nunique','count'],
                                                     'Occupation': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    channel_region_aggregate_features.columns = ['channel_region_aggregate_features' + '_'.join(c).strip('_') for c in channel_region_aggregate_features.columns]
    df = pd.merge(df, channel_region_aggregate_features, on = ['Channel_Code','Region_Code'], how='left')

    region_Gender_aggregate_features = df.groupby(['Region_Code','Gender']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Channel_Code': ['nunique','count'],
                                                     'Occupation': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    region_Gender_aggregate_features.columns = ['region_Gender_aggregate_features' + '_'.join(c).strip('_') for c in region_Gender_aggregate_features.columns]
    df = pd.merge(df, region_Gender_aggregate_features, on = ['Region_Code','Gender'], how='left')
    
    region_Occupation_aggregate_features = df.groupby(['Region_Code','Occupation']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Channel_Code': ['nunique','count'],
                                                     'Gender': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })

    region_Occupation_aggregate_features.columns = ['region_Occupation_aggregate_features' + '_'.join(c).strip('_') for c in region_Occupation_aggregate_features.columns]
    df = pd.merge(df, region_Occupation_aggregate_features, on = ['Region_Code','Occupation'], how='left')
   
    
    for i in cat_features:
        df[f'region_{i}_max']=df.groupby('Region_Code')[i].transform('max')
        df[f'region_{i}_min']=df.groupby('Region_Code')[i].transform('min')
        df[f'region_{i}_mean']=df.groupby('Region_Code')[i].transform('mean')
        df[f'region_{i}_std']=df.groupby('Region_Code')[i].transform('std')

    
        df[f'channel_region_{i}_max']=df.groupby(['Channel_Code','Region_Code'])[i].transform('max')
        df[f'channel_region_{i}_min']=df.groupby(['Channel_Code','Region_Code'])[i].transform('min')
        df[f'channel_region_{i}_mean']=df.groupby(['Channel_Code','Region_Code'])[i].transform('mean')
        df[f'channel_region_{i}_std']=df.groupby(['Channel_Code','Region_Code'])[i].transform('std')

    
        df[f'region_gender_{i}_max']=df.groupby(['Region_Code','Gender'])[i].transform('max')
        df[f'region_gender_{i}_min']=df.groupby(['Region_Code','Gender'])[i].transform('min')
        df[f'region_gender_{i}_mean']=df.groupby(['Region_Code','Gender'])[i].transform('mean')
        df[f'region_gender_{i}_std']=df.groupby(['Region_Code','Gender'])[i].transform('std')
        
    
    #features on gender
    
    gender_aggregate_features = df.groupby(['Gender']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Channel_Code': ['nunique','count'],       
                                                    'Region_Code': ['nunique','count'],
                                                     'Occupation': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    gender_aggregate_features.columns = ['gender_aggregate_features' + '_'.join(c).strip('_') for c in gender_aggregate_features.columns]
    df = pd.merge(df, gender_aggregate_features, on = ['Gender'], how='left')
        
        

    #features on Occupation 
    
    Occupation_aggregate_features = df.groupby(['Occupation']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Channel_Code': ['nunique','count'],       
                                                    'Region_Code': ['nunique','count'],
                                                     'Gender': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    Occupation_aggregate_features.columns = ['Occupation_aggregate_features' + '_'.join(c).strip('_') for c in Occupation_aggregate_features.columns]
    df = pd.merge(df, Occupation_aggregate_features, on = ['Occupation'], how='left')
    
    #Deriving characteristics of Accomodation_Type by creating aggregate features
    
    Channel_Code_aggregate_features = df.groupby(['Channel_Code']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Occupation': ['nunique','count'],       
                                                    'Region_Code': ['nunique','count'],
                                                     'Gender': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    Channel_Code_aggregate_features.columns = ['Channel_Code_aggregate_features' + '_'.join(c).strip('_') for c in Channel_Code_aggregate_features.columns]
    df = pd.merge(df, Channel_Code_aggregate_features, on = ['Channel_Code'], how='left')
    
    #Deriving characteristics of Interaction_features by creating aggregate features (These interaction feature are selected for aggregating based on its feature importance)
    
    Region_Code_grpd = df.groupby(['Region_Code']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Region_Code_grpd.columns = ['grpd_by_Region_Code_' + '_'.join(c).strip('_') for c in Region_Code_grpd.columns]
    df = pd.merge(df, Region_Code_grpd, on = ['Region_Code'], how='left')


    Channel_Code_grpd = df.groupby(['Channel_Code']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Channel_Code_grpd.columns = ['grpd_by_Channel_Code_' + '_'.join(c).strip('_') for c in Channel_Code_grpd.columns]
    df = pd.merge(df, Channel_Code_grpd, on = ['Channel_Code'], how='left')


    Occupation_grpd = df.groupby(['Occupation']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Occupation_grpd.columns = ['grpd_by_Occupation_' + '_'.join(c).strip('_') for c in Occupation_grpd.columns]
    df = pd.merge(df, Occupation_grpd, on = ['Occupation'], how='left')


    Gender_grpd = df.groupby(['Gender']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Gender_grpd.columns = ['grpd_by_Gender_' + '_'.join(c).strip('_') for c in Gender_grpd.columns]
    df = pd.merge(df, Gender_grpd, on = ['Gender'], how='left')
    
    
    Vintage_grpd = df.groupby(['Vintage']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Vintage_grpd.columns = ['grpd_by_Vintage_' + '_'.join(c).strip('_') for c in Vintage_grpd.columns]
    df = pd.merge(df, Vintage_grpd, on = ['Vintage'], how='left')


    Credit_Product_grpd = df.groupby(['Credit_Product']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Credit_Product_grpd.columns = ['grpd_by_Credit_Product_' + '_'.join(c).strip('_') for c in Credit_Product_grpd.columns]
    df = pd.merge(df, Credit_Product_grpd, on = ['Credit_Product'], how='left')



    Is_Active_grpd = df.groupby(['Is_Active']).agg({ 'Avg_Account_Balance': ['mean', 'max', 'min', 'std']})                                                              
                                                     
    Is_Active_grpd.columns = ['grpd_by_Is_Active_' + '_'.join(c).strip('_') for c in Is_Active_grpd.columns]
    df = pd.merge(df, Is_Active_grpd, on = ['Is_Active'], how='left')

    return df,le_features

"""### Remove unnecessary columns and prepare the train and test data for training"""

def preparedatafortraining(df,train,test):
    
    train=df.loc[df.train_or_test.isin(['train'])]
    test=df.loc[df.train_or_test.isin(['test'])]
    
    drop_columns={'ID','Is_Lead','train_or_test'}
    
    target=['Is_Lead']
    
    x=train.drop(columns=drop_columns,axis=1)
    y=train[target]
    x_test=test.drop(columns=drop_columns,axis=1)
    
    print(x.shape)
    
    return x,y,x_test

"""### Save Data"""

def savedata(**DATA_DIR):
    
    train,test,df=process_data("E:\FILES\VARIOUS PYTHON TASKS\AV LEAD")
    df,cat_features=feature_engineering(df)
    x_train,y_train,x_test=preparedatafortraining(df,train,test)
    
    #x_train.to_pickle("x_train_lgbm.pkl")
    #y_train.to_pickle("y_train_lgbm.pkl")
    #x_test.to_pickle("x_test_lgbm.pkl")
    
    return x_train,y_train,x_test,cat_features

"""### Train CatBoost Model and save the validation and test set prediction for ensembling"""

def catboost_model():
    
    x,y,x_test,cat_features=savedata()
     
    err = [] 

    oofs = np.zeros(shape=(len(x)))
    preds = np.zeros(shape=(len(x_test)))

    Folds=8

    fold = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=2021)
    i = 1

    for train_index, test_index in fold.split(x, y):
        x_train, x_val = x.iloc[train_index], x.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
        m =  CatBoostClassifier(n_estimators=10000,random_state=2021,eval_metric='AUC')
    
        m.fit(x_train, y_train,eval_set=[(x_val, y_val)], early_stopping_rounds=30,verbose=100,cat_features=cat_features)
    
        pred_y = m.predict_proba(x_val)[:,1]
        oofs[test_index] = pred_y
        print(i, " err_cat: ", roc_auc_score(y_val,pred_y))
        err.append(roc_auc_score(y_val,pred_y))
        preds+= m.predict_proba(x_test)[:,1]
        i = i + 1
        
        
    preds=preds/Folds
    
    print(f"Average StratifiedKFold Score : {sum(err)/Folds} ")
    oof_score = roc_auc_score(y, oofs)
    print(f'\nOOF Auc is : {oof_score}')
    
    oofs=pd.DataFrame(oofs,columns=['catboostoof'])
    preds=pd.DataFrame(preds,columns=['catboostpred'])
    
    oofs.to_csv('catboostoof.csv',index=False)
    preds.to_csv('catboostpred.csv',index=False)
    from datetime import datetime

    now = datetime.now()
    
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

catboost_model()
