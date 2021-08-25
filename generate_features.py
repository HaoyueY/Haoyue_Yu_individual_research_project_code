import os
import math
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython.display import Audio
from src.data import load_metadata, find_paths
from read_data import get_paths_and_starts,get_two_minute_features_opensmile,get_two_minute_features_yamnet,get_paths_and_starts2

opensmile_uri="/unix/cdtdisspotify/index/opensmile"
yamnet_embedding_uri="/unix/cdtdisspotify/index/yamnet/embedding"
yamnet_scores_uri="/unix/cdtdisspotify/index/yamnet/scores"
bm25_uri="/unix/cdtdisspotify/index/runs/UCL_pyserini_bm25.txt"
audio_uri='/unix/cdtdisspotify/data/spotify-podcasts-2020/podcasts-audio'


sample_rate=44100
RANDOM_SEED=0

raw_data=pd.read_csv(bm25_uri,sep='\t',header=None)
metadata=pd.read_csv("/unix/cdtdisspotify/data/spotify-podcasts-2020/metadata.tsv", delimiter="\t")
human_labeled=pd.read_csv('labeled.csv')

# Generate yamnet features for raw data
paths_yamnet_score, starts_yamnet_score=get_paths_and_starts(raw_data ,yamnet_scores_uri,'.h5')
scores_list = []
test_df=pd.read_hdf(paths_yamnet_score[0])
print(test_df.head())
for path, start in zip(tqdm(paths_yamnet_score), starts_yamnet_score):
    scores_list.append(get_two_minute_features_yamnet(path,start))
    
scores_df = pd.concat(scores_list,axis=0, ignore_index=True)
scores_df.to_csv('/unix/cdtdisspotify/haoyueyu/podcast-dataset/data/scores_df.csv',index=0)

# Generate yamnet features for human labeled data

paths,starts=get_paths_and_starts2(human_labeled,yamnet_scores_uri,'.h5')
scores_hl=[]
for path, start in zip(tqdm(paths), starts):
    scores_hl.append(get_two_minute_features_yamnet(path,start))
scores_h = pd.concat(scores_hl,axis=0, ignore_index=True)
scores_h.to_csv('/unix/cdtdisspotify/haoyueyu/podcast-dataset/data/scores_hl.csv',index=0)


# Generate eGeMAPS features for raw data
paths_opensmile, starts_opensmile=get_paths_and_starts(raw_data,opensmile_uri,'.h5')
functionals_list = []
test_df=pd.read_hdf(paths_opensmile[0]).head()
#get the column dictionary to compute 2 minute features
col_dic=get_col_dictionary(test_df.columns)
for path, start in zip(tqdm(paths_opensmile), starts_opensmile):
    functionals_list.append(get_two_minute_features(path,start,col_dic)) 
functionals_df = pd.concat(functionals_list,axis=0, ignore_index=True)
functionals_df.to_csv('/unix/cdtdisspotify/haoyueyu/podcast-dataset/data/functionals_df.csv',index=0)

# Generate eGeMAPS features for human labeled data
paths,starts=get_paths_and_starts2(human_labeled,opensmile_uri,'.h5')
functionals_hl=[]
for path, start in zip(tqdm(paths), starts):
    functionals_hl.append(get_two_minute_features(path,start,col_dic))
functionals_h = pd.concat(functionals_hl,axis=0, ignore_index=True)
functionals_h.to_csv('/unix/cdtdisspotify/haoyueyu/podcast-dataset/data/functionals_h.csv',index=0)