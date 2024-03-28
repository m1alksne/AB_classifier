
"""
Created on Mon Jan 23 10:35:46 2024

@author: Michaela Alksne

Script to make train, validation, and test datasets. 
First we decide how to split up the data we have available: 
    -Training dataset: used to fit the model to the audio data
    -Validation dataset: used to evaluate model preformance and tune hyper-parameters 
    -Test dataset: used after the model has completed training to evaluate model preformance on unseen/novel data. Best to have this data represent a new "domain" 
    
    - In this example, we have two annotated datasets: DCPP01A and SOCAL34N
    - Here we split DCPP01A into train and validation based on the distribution of calls in each xwav file
    - SOCAL34N is used for testing 
    
We read in our annotations and generate "one_hot_clips" for our audio files
    - each row represents a single sample, in our case, a 30 second audio clip
    - the first column "file" contains the path to our audio file. 
    - The second column "start_time" contains the start time of the audio clip, in seconds since the start of the xwav
    - The third column "end_time" contains the end time of the audio clip, in seconds since the start of the xwav 
    - The next columns represent the possible classes: "A NE Pacific", "B NE Pacific"
    - A "0" means that in that sample, the class is not present
    - A "1" means that in that sample, the class is present
    
Here, we generate "one_hot_clips" from each of our datasets using the following parameters:
    - clip_duration: 30 seconds. This is the length of each audio segment
    - clip_overlap: 10 seconds. We step ahead every ten seconds, this effectively creates more unique samples for the model to draw from
    - min_label_overlap: 5 seconds. This is the minimum duration of a call that must be in a given clip for it to be labeled as "present" 
    
Next, we balance each of our datasets. This is especially important for our training dataset to avoid biases, improve generalization, and reduce overfitting. 
    - We include 1500 examples from each class for our training dataset
    - We also make sure to include 1500 examples that do not contain either class to improve discrimination between target and non-target images or patterns
    
Lastly, we save our newly created train, validation, and test clips for the next step (which is training the model!)
    
"""
import opensoundscape
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import librosa
import torch
import random
import sys

# Add the parent directory to sys.path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# import config values
from config import repo_path

# read in datasets

DCPP01A = pd.read_csv(repo_path/'labeled_data'/'logs'/'modified_annotations'/'DCPP01A_logs_modification.csv')
CINMS18B = pd.read_csv(repo_path/'labeled_data'/'logs'/'modified_annotations'/'CINMS18B_logs_modification.csv')

# Filter rows for training set where 'audio_file' column does not equal 'DCPP01A_d01_121106_083945.d100.x.wav'
train_annotations = DCPP01A[~DCPP01A['audio_file'].str.contains('DCPP01A_d01_121106_083945.d100.x.wav')]
# Filter rows for validation set where 'audio_file' column contains 'DCPP01A_d01_121106_083945.d100.x.wav'
val_annotations = DCPP01A[DCPP01A['audio_file'].str.contains('DCPP01A_d01_121106_083945.d100.x.wav')]

test_annotations = CINMS18B

# count up all the calls
train_call_counts = train_annotations['annotation'].value_counts()
print(train_call_counts)
val_call_counts = val_annotations['annotation'].value_counts()
print(val_call_counts)
test_call_counts = test_annotations['annotation'].value_counts()
print(test_call_counts)

# make boxed annotations
# converts the annotations to opensoundscape package format
train_annotations_box = opensoundscape.BoxedAnnotations(train_annotations)
val_annotations_box = opensoundscape.BoxedAnnotations(val_annotations)
test_annotations_box = opensoundscape.BoxedAnnotations(test_annotations)

# make hot clips
train_clips = train_annotations_box.one_hot_clip_labels(audio_files=train_annotations_box.df['audio_file'].unique(), clip_duration=30, clip_overlap=10, min_label_overlap=5, class_subset=['A NE Pacific', 'B NE Pacific'])
print(train_clips.sum())
val_clips = val_annotations_box.one_hot_clip_labels(audio_files=val_annotations_box.df['audio_file'].unique(), clip_duration=30,clip_overlap=10, min_label_overlap=5, class_subset=['A NE Pacific', 'B NE Pacific'])
test_clips = test_annotations_box.one_hot_clip_labels(audio_files=test_annotations_box.df['audio_file'].unique(), clip_duration=30, clip_overlap=10, min_label_overlap=5, class_subset=['A NE Pacific', 'B NE Pacific'])
balanced_train_clips = opensoundscape.data_selection.resample(train_clips,n_samples_per_class=1000,random_state=0) # upsample (repeat samples) so that all classes have 1000 samples
balanced_val_clips = opensoundscape.data_selection.resample(val_clips,n_samples_per_class=200,random_state=0) # upsample (repeat samples) so that all classes have 400 samples

# add empty spectrograms to training data so the model learns what is "not" a call
train_clips_new = train_clips.reset_index(drop=True) # drop index to allow for indexing in next line
noise_indices = train_clips_new.index[(train_clips['A NE Pacific'] == 0) & (train_clips['B NE Pacific'] == 0)]# indices of negatives (times when there are no calls)
random_noise_indices = random.sample(noise_indices.tolist(), 1000) # random sample subset of the empty examples
train_clips_noise = train_clips.iloc[random_noise_indices] # subset by these indices
train_clips_final = pd.concat([balanced_train_clips, train_clips_noise]) # concatenate dataframes

# save dataframes
train_clips_final.to_csv(repo_path/'labeled_data'/'train_val_test_clips'/'train_clips.csv')
balanced_val_clips.to_csv(repo_path/'labeled_data'/'train_val_test_clips'/'val_clips.csv')
test_clips.to_csv(repo_path/'labeled_data'/'train_val_test_clips'/'test_clips.csv')