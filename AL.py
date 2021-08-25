import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn import metrics
import random
from read_data import id_to_path_and_start, get_audio_segment
import soundfile as sf
from IPython.display import Audio
import IPython

bm25_uri="/unix/cdtdisspotify/index/runs/UCL_pyserini_bm25.txt"
audio_uri='/unix/cdtdisspotify/data/spotify-podcasts-2020/podcasts-audio'


metadata=pd.read_csv("/unix/cdtdisspotify/data/spotify-podcasts-2020/metadata.tsv", delimiter="\t")
raw_data=pd.read_csv(bm25_uri,sep='\t',header=None)


def select_query(X,model,df_cluster):
    # Select the query from unlabeled  data
    #
    # Step 1: Select most uncertain data with replace
    y_unlabeled=model.predict_proba(X)
    confidence_df=pd.DataFrame(index=X.index,columns=['probability'])
    confidence_df['probability']=model.predict_proba(X).max(axis=1)
    confidence_df['pct']=confidence_df.rank(pct=True)
    selected_index1=confidence_df[(confidence_df['pct']>=0) & (confidence_df['pct']<0.1)].index

    # Step 2: use DBSCAN to find the representative data, avoid outlier
    #
    #
    #             cluster: -1 is noise
    temp_df=df_cluster.loc[selected_index1,:]
    selected_index2=temp_df[temp_df.cluster_db!=-1].index
    if len(selected_index2)>=1:
        selected_index2=random.sample(selected_index2.tolist(), 1)
    return selected_index2

def play_audio(index,df,audio_path):
    # play selected audio by get the waveform
    p,s = id_to_path_and_start(df.iloc[index,2],audio_path,'.ogg')
    waveform=get_audio_segment(p,s)
    return waveform

def generate_evaluation_dictionary(y,y_pred,y_score):
    # generate evalution dictionary
    output_dict=dict()
    output_dict['accuracy']=metrics.accuracy_score(y,y_pred)
    output_dict['roc_auc']=metrics.roc_auc_score(y,y_score[:,1])
    output_dict['f1']=metrics.f1_score(y,y_pred)
    output_dict['precision']=metrics.precision_score(y,y_pred,average='weighted')
    output_dict['recall']=metrics.recall_score(y,y_pred)
    return output_dict

def run_grid_search_cv( X_train,y_train,model,param_grid):
    # run grid search cross validation
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(X_train,y_train)
    clf=grid.best_estimator_
    return clf


def run( N_query, X_train, y_train,df_cluster, clf,test_data,unlabeled,evaluation_df):
    X_train_update=X_train.copy()
    y_train_update=y_train.copy()
    unlabeled_update=unlabeled.copy()
    for i in range(N_query):
        print('Query %d starts!' %i)
        # select the query
        query_index = select_query(unlabeled_update,clf,df_cluster)
        # load selected data and label
        for ii in query_index:
            wf=play_audio(ii,raw_data,audio_uri)
            IPython.display.display(Audio(wf, rate=44100))
            label=int(input('label the segment you listened ( 1 for funny, -1 for not funny) : '))
            X_train_update=X_train_update.append(unlabeled.iloc[ii,:], ignore_index=False)
            y_train_update=y_train_update.append(pd.Series(label,index=[ii]),ignore_index=False)
        # teach the model
        clf=clf.fit(X_train_update,y_train_update)
        y_predict=clf.predict(test_data.iloc[:,1:])
        y_score=clf.predict_proba(test_data.iloc[:,1:])
        evaluation_dic=generate_evaluation_dictionary(test_data.label, y_predict,y_score)
        evaluation_df=evaluation_df.append(pd.DataFrame(evaluation_dic,index=[i+1]))
        print('accuracy is : %s' %evaluation_dic['accuracy'])
        # Remove data from unlabeled data
        unlabeled_update.drop(query_index,inplace=True)
        
    return X_train_update,y_train_update,clf,unlabeled_update,evaluation_df



