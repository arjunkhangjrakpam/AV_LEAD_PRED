

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', 100)
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

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
    cat_features=[]
    columns=['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code','Vintage', 'Credit_Product',  'Is_Active']

    comb = combinations(columns, 2) 

    for i in list(comb):  
        df[f'{i[0]}_{i[1]}']=df[i[0]].astype(str)+'_'+df[i[1]].astype(str)     
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
        
     #Creating Age Bins and deriving characteristics of each age group by creating aggregate features
    
    Age_Bins = KBinsDiscretizer(n_bins=9, encode='ordinal', strategy='quantile')
    df['Age_Bins'] =Age_Bins.fit_transform(df['Age'].values.reshape(-1,1)).astype(int)
    
    age_aggregate_features = df.groupby(['Age_Bins']).agg({'Age': ['mean', 'max', 'min','std'],
                                                    'Region_Code': ['nunique','count'], 
                                                    'Channel_Code': ['nunique','count'],       
                                                    'Gender': ['nunique','count'],
                                                     'Occupation': ['nunique','count'] ,
                                                     'Vintage': ['nunique','count'] ,
                                                     'Credit_Product': ['nunique','count'] ,
                                                     'Is_Active': ['nunique','count'] ,
                                                     })
    age_aggregate_features.columns = ['age_aggregate_features' + '_'.join(c).strip('_') for c in age_aggregate_features.columns]
    df = pd.merge(df, age_aggregate_features, on = ['Age_Bins'], how='left')
    
    
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
    
    return df,cat_features

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
    
    return x_train,y_train,x_test

    
def score(params):
    try:

        print("Training with params: ",params)
        num_round = int(params['n_estimators'])
        del params['n_estimators']
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
        gbm_model = xgb.train(params, dtrain, num_round,
                              evals=watchlist,
                              verbose_eval=False)
        predictions = gbm_model.predict(dvalid,
                                        ntree_limit=gbm_model.best_iteration + 1)
        predictions = (predictions >= 0.5).astype('int')
        score = f1_score(y_valid, predictions, average='weighted')
        print("\tScore {0}\n\n".format(score))
        
        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {'loss': loss, 'status': STATUS_OK}
   
    # In case of any exception or assertionerror making score 0, so that It can return maximum loss (ie 1)
    except AssertionError as obj:
        #print("AssertionError: ",obj)
        loss = 1 - 0
        return {'loss': loss, 'status': STATUS_OK}

    except Exception as obj:
        #print("Exception: ",obj)
        loss = 1 - 0
        return {'loss': loss, 'status': STATUS_OK}


def optimize(
             trials, 
             max_evals, 
             random_state= 1):


    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 15, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.05),
        'gamma': hp.quniform('gamma', 0, 1, 0.05),
        'learning_rate': hp.quniform('learning_rate', 0, 1, 0.001),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'scale_pos_weight': hp.quniform('scale_pos_weight', 1,4, 0.05),
        "alpha": hp.quniform('alpha', 0.0001, 1, 0.005),
        "lambda": hp.quniform('lambda', 1, 5, 0.05),
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default 
        # to the maxium number. 
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, 
                space, 
                algo=tpe.suggest, 
                trials=trials, 
                max_evals=max_evals)
    return best



""" Get the best hyperparameters """
def getparams(x, y):
    
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, random_state=45, test_size=0.2)
    
    trials = Trials()
    MAX_EVALS = 50
    
    best_hyperparams = optimize(trials, MAX_EVALS)
    print("The best hyperparameters are: ", "\n")
    print(best_hyperparams)
    
    return best_hyperparams


"""### Train LGBM Model and save the validation and test set prediction for ensembling"""

def lgbm_model():
    
    x,y,x_test=savedata()
    
    params={'lambda': 2.8849054495567423, 
        'alpha': 0.001054193185317787, 
        'colsample_bytree': 0.5, 
        'subsample': 0.4, 
        'learning_rate': 0.014, 
        'max_depth': 13, 
        'random_state': 24,
        'min_child_weight': 5}
    
    err = [] 

    oofs = np.zeros(shape=(len(x)))
    preds = np.zeros(shape=(len(x_test)))

    Folds=8

    fold = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=2020)
    i = 1

    for train_index, test_index in fold.split(x, y):
        x_train, x_val = x.iloc[train_index], x.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
        m = LGBMClassifier(n_estimators=10000,**params,verbose= -1)
    
        m.fit(x_train, y_train,eval_set=[(x_val, y_val)], early_stopping_rounds=30,verbose=False,eval_metric='auc')
    
        pred_y = m.predict_proba(x_val)[:,1]
        oofs[test_index] = pred_y
        print(i, " err_lgm: ", roc_auc_score(y_val,pred_y))
        err.append(roc_auc_score(y_val,pred_y))
        preds+= m.predict_proba(x_test)[:,1]
        i = i + 1
    preds=preds/Folds
    
    print(f"Average StratifiedKFold Score : {sum(err)/Folds} ")
    oof_score = roc_auc_score(y, oofs)
    print(f'\nOOF Auc is : {oof_score}')
    
    oofs=pd.DataFrame(oofs,columns=['lgbmoof'])
    preds=pd.DataFrame(preds,columns=['lgbmpred'])
    
    oofs.to_csv('lgbmoof.csv',index=False)
    preds.to_csv('lgbmpred.csv',index=False)
    
    from datetime import datetime

    now = datetime.now()
    
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

lgbm_model()

"""### Train XGB Model and save the validation and test set prediction for ensembling"""

def xgb_model():
    
    x,y,x_test=savedata()
       
    params = getparams(x, y)
    
    err = [] 

    oofs = np.zeros(shape=(len(x)))
    preds = np.zeros(shape=(len(x_test)))

    Folds=8

    fold = StratifiedKFold(n_splits=Folds, shuffle=True, random_state=2020)
    i = 1

    for train_index, test_index in fold.split(x, y):
        x_train, x_val = x.iloc[train_index], x.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
        m = XGBClassifier(n_estimators=10000,**params)
    
        m.fit(x_train, y_train,eval_set=[(x_val, y_val)], early_stopping_rounds=30,verbose=False,eval_metric='auc')
    
        pred_y = m.predict_proba(x_val)[:,1]
        oofs[test_index] = pred_y
        print(i, " err_xgb: ", roc_auc_score(y_val,pred_y))
        err.append(roc_auc_score(y_val,pred_y))
        preds+= m.predict_proba(x_test)[:,1]
        i = i + 1
    preds=preds/Folds
    
    print(f"Average StratifiedKFold Score : {sum(err)/Folds} ")
    oof_score = roc_auc_score(y, oofs)
    print(f'\nOOF Auc is : {oof_score}')
    
    oofs=pd.DataFrame(oofs,columns=['xgboof'])
    preds=pd.DataFrame(preds,columns=['xgbpred'])
    
    oofs.to_csv('xgbmoof.csv',index=False)
    preds.to_csv('xgbmpred.csv',index=False)
    
    from datetime import datetime

    now = datetime.now()
    
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

xgb_model()
