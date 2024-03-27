# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:08:32 2024

@author: Michaela Alksne

This script allows a user to evaluate model training preformance on train and validation datasets. 
Here we generate predictions for train and validation datasets from each epoch of the model. 
This allows us to visualize how model preformance improves/changes over each epoch. 
Here we plot mean average precision, which is the area under a precision-recall curve. 
This is a commonly used metric to evaluate preformance. 
During training, loss is computed to guide backpropagation and adjust weights, and mean average precision is computed and used to select the best model. 
This script allows the user to plot mean average precision per epoch after training is complete. 
This is simply a visualization tool and allows you to understand your model better.

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
import wandb
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score

from config import repo_path, xwavs_path

train_clips = pd.read_csv(repo_path/'labeled_data'/'train_val_test_clips'/'train_clips.csv')
train_clips['file'] = str(xwavs_path) + '/' + train_clips['file']
train_clips.set_index(['file', 'start_time', 'end_time'], inplace=True)

val_clips = pd.read_csv(repo_path/'labeled_data'/'train_val_test_clips'/'val_clips.csv')
val_clips['file'] = str(xwavs_path) + '/' + val_clips['file']
val_clips.set_index(['file', 'start_time', 'end_time'], inplace=True)

# Load train and validation datasets

# Initialize lists to store metrics for each epoch
train_ap_A_history = []
train_ap_B_history = []
val_ap_A_history = []
val_ap_B_history = []

# Iterate over each epoch of the saved model
for epoch in range(11):  # assuming you have 11 epochs
    # Load the model for the current epoch
    model_path = repo_path/'train'/'model_states'/f'epoch-{epoch}.model'
    model = opensoundscape.ml.cnn.load_model(model_path)
    
    # Make predictions on the training dataset
    train_scores = model.predict(train_clips, num_workers=12, batch_size=128)
    train_scores.columns = ['pred_A','pred_B']
    train_all = train_clips.join(train_scores)
    
    # Make predictions on the validation dataset
    val_scores = model.predict(val_clips, num_workers=12, batch_size=128)
    val_scores.columns = ['pred_A','pred_B']
    val_all = val_clips.join(val_scores)
    
    # Calculate and store Average Precision (AP) for training and validation datasets
    train_ap_A = average_precision_score(train_all['A NE Pacific'], train_all['pred_A'])
    train_ap_B = average_precision_score(train_all['B NE Pacific'], train_all['pred_B'])
    
    val_ap_A = average_precision_score(val_all['A NE Pacific'], val_all['pred_A'])
    val_ap_B = average_precision_score(val_all['B NE Pacific'], val_all['pred_B'])

    train_ap_A_history.append(train_ap_A)
    train_ap_B_history.append(train_ap_B)
    
    val_ap_A_history.append(val_ap_A)
    val_ap_B_history.append(val_ap_B)
    

plt.subplot(1, 2, 2)
plt.plot(range(epoch+1), train_ap_A_history, label='Training Average Precision A calls')
plt.plot(range(epoch+1), train_ap_B_history, label='Training Average Precision B calls')
plt.xlabel('Epoch')
plt.ylabel('Average precision over epochs')
plt.legend()
plt.title('Average Precision Train')
    
plt.tight_layout()
plt.show()
    
plt.subplot(1, 2, 2)
plt.plot(range(epoch+1), val_ap_A_history, label='Validation Average Precision A calls')
plt.plot(range(epoch+1), val_ap_B_history, label='Validation Average Precision B calls')
plt.xlabel('Epoch')
plt.ylabel('Average precision over epochs')
plt.legend()
plt.title('Average precision Validation')
    
plt.tight_layout()
plt.show()



