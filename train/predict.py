"""
Created on Mon Jan 22 11:35:46 2024

@author: Michaela Alksne

Script to use the model for running inference, or predicting on new data.
Our model has a "predict" function which we can call to predict on a new dataset. 
In this case, we are predicting on our test data. Therefore we are able to generate preformance metrics as described below. 
However, if we were predicting on unlabeled data, we would just the use the models predict function and would not be able to plot the preformance metrics.

Here we load in our trained model and modify the spectrogram parameters because our test data has a different sampling rate than our training data. This will not effect the model. This is just resizing the images so they match. 
Our model has a "predict" function which we can call to predict on a new dataset. 
    - We load in our model and our test data and generate predictions. 
    - Then we join the predictions with the true labels and evaluate model preformance by plotting our precision-recall curve using scikit learn
    - We also plot the distribution of our scores for true and false detections. 

"""
import opensoundscape
import glob
import os
import pandas as pd
import numpy as np
import sklearn
import librosa
import torch
import wandb
import random
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# read in train and validation dataframes
test_clips = pd.read_csv(repo_path/labeled_data/train_val_test_clips/test_clips.csv', index_col=[0,1,2]) 
print(test_clips.sum())

model = opensoundscape.ml.cnn.load_model(repo_path/train\model_states/best.model')

# moodify model preprocessing for making spectrograms with proper resolution
model.preprocessor.pipeline.to_spec.params.window_type = 'hamming' # using hamming window
model.preprocessor.pipeline.to_spec.params.window_samples = 1000 # window samples
model.preprocessor.pipeline.to_spec.params.overlap_samples = 900 # 90% overlap, for 3200 Fs this means 900 samples, and 0.05 sec bins
model.preprocessor.pipeline.to_spec.params.fft_size = 2000 # FFT = Fs, 1 Hz bins
model.preprocessor.pipeline.to_spec.params.decibel_limits = (-120,150) # oss preprocessing sets dB limits.

# predict 
test_scores = model.predict(test_clips, num_workers=12,batch_size=128)
test_scores.columns = ['pred_A','pred_B']
test_all = test_clips.join(test_scores)
#save output 
test_all.to_csv(repo_path/labeled_data/train_val_test_clips/test_clips_prediction.csv')

## A CALLS ###

# plot precision recall curve for A calls
precision, recall, thresholds = precision_recall_curve(test_all['A NE Pacific'], test_all['pred_A'])
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
#add axis labels to plot
ax.set_title('Precision-Recall Curve A calls test data')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
#display plot
plt.show()

# plot score distribution A calls 
A_eval_index = test_all.index[test_all['A NE Pacific']==1]
A_eval = test_all.loc[A_eval_index]
A_noise_index = test_all.index[test_all['A NE Pacific']==0]
A_noise = test_all.loc[A_noise_index]
plt.hist(A_noise['pred_A'],bins=40,alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(A_eval['pred_A'],bins=40,alpha=0.5,edgecolor='black',color='orange',label='A call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('A call prediction scores val data')
plt.legend(loc='upper right')


## B CALLS ###

# plot precision recall curve for B calls
precision, recall, thresholds = precision_recall_curve(test_all['B NE Pacific'], test_all['pred_B'])
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
#add axis labels to plot
ax.set_title('Precision-Recall Curve B calls test data')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
#display plot
plt.show()

# plot score distribution B calls 
B_eval_index = test_all.index[test_all['B NE Pacific']==1]
B_eval = test_all.loc[B_eval_index]
B_noise_index = test_all.index[test_all['B NE Pacific']==0]
B_noise = test_all.loc[B_noise_index]
plt.hist(B_noise['pred_B'],bins=40,alpha=0.5,edgecolor='black',color='blue',label='Noise prediction score')
plt.hist(B_eval['pred_B'],bins=40,alpha=0.5,edgecolor='black',color='orange',label='B call prediction score')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.semilogy()
plt.title('B call prediction scores val data')
plt.legend(loc='upper right')