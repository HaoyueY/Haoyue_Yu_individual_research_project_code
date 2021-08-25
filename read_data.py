import os
import math
from pathlib import Path
import IPython
import pandas as pd
import numpy as np
from tqdm import tqdm
import soundfile as sf
from IPython.display import Audio
from src.data import load_metadata, find_paths
import random


metadata=pd.read_csv("/unix/cdtdisspotify/data/spotify-podcasts-2020/metadata.tsv", delimiter="\t")
sample_rate=44100


def id_to_path_and_start(id,path,file_type):
    """Get the opensmile file path and start for a given segment id."""
    episode_uri, segment_start = id.split("_")
    episode = metadata[(metadata.episode_uri == episode_uri).values]
    files = find_paths(episode, path, file_type)
    return files[0], int(float(segment_start))

def get_paths_and_starts(segments,rawpath,appendix):
    """Get the opensmile file paths for the given segments."""
    paths = []
    starts = []
    for segment in segments[2]:
        path, start = id_to_path_and_start(segment,rawpath,appendix)
        paths.append(path)
        starts.append(start)
    return paths, starts

def get_audio_segment(path, start, duration=120):    
    """Get the required podcast audio segment."""
    waveform, sr = sf.read(
        path,
        start=start * sample_rate,
        stop=(start + duration) * sample_rate,
        dtype=np.int16,
    )
    waveform = np.mean(waveform, axis=1) / 32768.0
    return waveform

def get_col_dictionary(columns):
    col_dict=dict()
    col_dict['std_cols']=[std_col for std_col in columns if 'stddevNorm' in std_col or 'Stddev' in std_col]
    col_dict['mean_cols']=[mean_col for mean_col in columns if 'mean' in mean_col or 'Mean' in mean_col or 'Slope' in mean_col]
    col_dict['min_cols']=[min_col for min_col in columns if 'percentile20' in min_col]
    col_dict['max_cols']=[max_col for max_col in columns if 'percentile80' in max_col or 'Peaks' in max_col]
    col_dict['diff_cols']=[diff_col for diff_col in columns if 'pctlrange' in diff_col]
    col_dict['rest_cols']=[rest_col for rest_col in columns if rest_col not in col_dict['std_cols'] and rest_col not in col_dict['mean_cols'] and rest_col not in col_dict['min_cols'] and rest_col not in col_dict['max_cols'] and rest_col not in col_dict['diff_cols']]
    return col_dict
    
def get_two_minute_features_opensmile(path, start, col_dic):
    # 2 minutes segment with 0.48s per sample
    temp_df=pd.read_hdf(path).iloc[start:start+250,:]
    output_df=pd.DataFrame(columns=temp_df.columns,index=temp_df.index[0:1])
    
    # Compute standard deviatin 
    for i in col_dic['std_cols']:
        output_df[i]=np.sqrt(np.mean(temp_df[i]**2+temp_df.iloc[:,temp_df.columns.get_loc(i)-1]**2)-temp_df.iloc[:,temp_df.columns.get_loc(i)-1].mean()**2)
    
    # Compute mean 
    for j in col_dic['mean_cols']:
        output_df[j]=np.mean(temp_df[j])
    
    # Compute minimum
    for k in col_dic['min_cols']:
        output_df[k]=np.min(temp_df[k])

    # Compute maximum
    for l in col_dic['max_cols']:
        output_df[l]=np.max(temp_df[l])
    
    # Compute difference
    for m in col_dic['diff_cols']:
        output_df[m]=output_df.iloc[:,temp_df.columns.get_loc(m)-1]-output_df.iloc[:,temp_df.columns.get_loc(m)-3]
        
    # Take mean for the rest data
    for n in col_dic['rest_cols']:
        output_df[n]=np.mean(temp_df[n])
    
    return output_df


def get_two_minute_features_yamnet(path, start):
    # 2 minutes segment with 0.48s per sample
    # Compute sum
    temp_df=pd.read_hdf(path).iloc[start:start+250,:]
    output_df=pd.DataFrame(temp_df.sum(axis=0)).T
    return output_df



def get_paths_and_starts2(df,path,file_type):
    paths=[]
    starts=[]
    for i,r in df.iterrows():
        episode = metadata[(metadata.episode_uri == r['uri']).values]
        file = find_paths(episode, path, file_type)[0]
        paths.append(file)
        starts.append(r['timestamp'])
    return paths, starts