# UCL_Dissertation
The code should be run on UCL's HEP machine


## Initialising
Import the Spotify Podcast Dataset software and intall dependencies/src from https://github.com/ucl-dis-spotify-group-project/podcast-dataset

## Loading data
The python script read_data.py includes all of the functions needed to generate feature data.
The script generate_features.py is used to generate features data, the generated features are saved on the hep machine in the folder: '/unix/cdtdisspotify/haoyueyu/podcast-dataset/data/'

## Run models:
Opensmile_baseline.ipynb and Yamnet_baseline.ipynb are notebooks used to run the baseline models
AL.py is the script for all of the functions used in active learning
ActiveLearning.ipynb is the notebook for running active learning
