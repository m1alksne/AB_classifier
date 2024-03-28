
"""
Created on Mon Jan 22 11:35:46 2024

@author: Michaela Alksne

Script to train a resnet-18 CNN to classify A and B calls in 30 second spectrograms
sets model and spectrogram parameters and connects to wandB so user can monitor training progress

Model parameters: 
    - multi-target model: 3 labels per sample
    - classification with ResampleLoss function
    - weights pretrained on ImageNet
    - learning rate = 0.001
    - cooling factor = 0.3 (decreases learning rate by multiplying 0.001*3 every ten epochs)
    - epochs = 12 
    - batch_size = 12

Spectrogram parameters:
    - 30 second windows
    - 3200 Hz(samples/second) sampling rate 
    - 3200 point-FFT which results in 1 Hz bins
    - 90 % overlap (or 1400 samples), resulting in 0.05 second bins
    - 1600 Hamming window samples. A Hamming window is used to smooth the signal and reduce spectral leakage/artifacts for the FFT. 
    - minimum frequency: 10 Hz
    - maximum frequency: 150 Hz
    
Spectrogram augmentations: 
    - frequency_mask: adds random horizontal bars over image
    - time_mask: adds random vertical bars over the image
    - add_noise: adds random Gaussian noise to image 
    
Notes for user:
batch_size – number of training files to load/process before re-calculating the loss function and backpropagation
num_workers – parallelization (ie, cores or cpus)
log_interval – interval in epochs to evaluate model with validation dataset and print metrics to the log

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

# read in train and validation dataframes
train_clips = pd.read_csv('L:\\HARP_CNN\\AB_classifier\\labeled_data\\train_val_test_clips\\train_clips.csv', index_col=[0,1,2]) 
val_clips = pd.read_csv('L:\\HARP_CNN\\AB_classifier\\labeled_data\\train_val_test_clips\\val_clips.csv', index_col=[0,1,2]) 
print(train_clips.sum()) 
print(val_clips.sum())

calls_of_interest = ["A NE Pacific", "B NE Pacific"] #define the calls for CNN
model = opensoundscape.CNN('resnet18',classes=calls_of_interest,sample_duration=30.0, single_target=False) # create a CNN object designed to recognize 30-second samples
opensoundscape.ml.cnn.use_resample_loss(model) # loss function for mult-target classification

# moodify model preprocessing for making spectrograms the way I want them
model.preprocessor.pipeline.to_spec.params.window_type = 'hamming'
model.preprocessor.pipeline.to_spec.params.window_samples = 1600 
model.preprocessor.pipeline.to_spec.params.overlap_samples = 1400 
model.preprocessor.pipeline.to_spec.params.fft_size = 3200 
model.preprocessor.pipeline.to_spec.params.decibel_limits = (-120,150)
model.preprocessor.pipeline.to_spec.params.scaling = 'density'
model.preprocessor.pipeline.bandpass.params.min_f = 10
model.preprocessor.pipeline.bandpass.params.max_f = 150
model.preprocessor.pipeline.frequency_mask.bypass = True
#model.preprocessor.pipeline.time_mask.bypass = True
#model.preprocessor.pipeline.frequency_mask.set(max_width = 0.003, max_masks=1)
model.preprocessor.pipeline.time_mask.set(max_width = 0.1, max_masks=5)
model.preprocessor.pipeline.add_noise.set(std=0.1)
model.preprocessor.pipeline.random_affine.bypass=True
model.optimizer_params['lr']=0.001
model.lr_cooling_factor = 0.3 
model.wandb_logging['n_preview_samples']=100 # number of samples to look at in wandB


wandb_session = wandb.init( #initialize wandb logging 
        entity='BigBlueWhale', #replace with your entity/group name
        project='Sonobuoy Model',
        name='Trial 9: 30 second windows')

model.train(
    train_clips, 
    val_clips, 
    epochs = 12, 
    batch_size= 128, 
    log_interval=1, #log progress every 1 batches
    num_workers = 12, 
    #wandb_session=wandb_session,
    save_interval = 1, #save checkpoint every 1 epoch
    save_path = 'L:\\HARP_CNN\\AB_classifier\\train\\model_states' #location to save checkpoints (epochs)
)

