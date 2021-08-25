# UCL_Dissertation
The code should be run on UCL's HEP machine


## Initialising
Import the Spotify Podcast Dataset software and intall dependencies/src from: https://github.com/ucl-dis-spotify-group-project/podcast-dataset

## Modules
This section list the necessary modules that are need to be loaded before implementation

· The python script read_data.py includes all of the functions needed to generate feature data.
The script generate_features.py is used to generate features data, the generated features are saved on the hep machine in the folder: '/unix/cdtdisspotify/haoyueyu/podcast-dataset/data/'
AL.py is the script for all of the functions used in active learning

## Implementation Notebooks:
Opensmile_baseline.ipynb and Yamnet_baseline.ipynb are notebooks used to run the baseline models
ActiveLearning.ipynb is the notebook for running active learning

## Data:
Because the raw datasets are too large to be uploaded, here provide a subset of all the datasets that cannot be uploaded: functionals_df(sample).csv; raw_data(sample).csv; scores_df(sample).csv
The human labeled data and their features are completely uploaded.
A sample of 2-min audio is also provided.
