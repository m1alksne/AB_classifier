# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:20:28 2024

@author: Michaela Alksne 

First script to run when modifying Triton logger annotation excel datasheets
converts xls to csv containing the audio file path, the annotation label, the frequency bounds, and time bounds. 
saves new csv in "modified annotations subfolder"
xwav is the audio file
start time = start time of call in number of seconds since start of xwav 
end time = end time of call in number of seconds since start of xwav

"""

from datetime import datetime
import os
import glob
import opensoundscape
import sys
from opensoundscape import Audio, Spectrogram
import random
import pandas as pd
import numpy as np
import xlrd

# Add the parent directory to sys.path
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
#sys.path.append("L:\HARP_CNN\AB_classifier\AB_classifier")
# import config values
from config import repo_path, xwavs_path

sys.path.append(str(repo_path / "preprocessing"))
from AudioStreamDescriptor import XWAVhdr

directory_path = repo_path / 'labeled_data' / 'logs'  # point to original logger files
all_files = glob.glob(os.path.join(directory_path, '*.xls'))  # path for all files

#new_base_path = str(xwavs_path) + '/'  # path to change to
new_base_path = str(xwavs_path) + '\\'  # path to change to


# hepler function uses XWAVhdr to read xwav file header info and extract xwav file start time as a datetime object
def extract_wav_start(path):
    xwav_hdr = XWAVhdr(path)
    xwav_start_time = xwav_hdr.dtimeStart
    return xwav_start_time


# helper function to modify the original logger files to new format
# replaces xwav file path and converts start and end time to seconds since start of xwav file.
def modify_annotations(df):
    # df['audio_file'] = [in_file.replace(os.path.split(in_file)[0], new_base_path) for in_file in df['Input file']] # uses list comprehension to replace old wav path with new one
    # this should work between Windows and Linux:
    df['audio_file'] = df['Input file'].apply(lambda x: new_base_path + x.split('\\')[-1])

    df['file_datetime'] = df['audio_file'].apply(
        extract_wav_start)  # uses .apply to apply extract_wav_start time from each wav file in the list
    df['start_time'] = (df['Start time'] - df[
        'file_datetime']).dt.total_seconds()  # convert start time difference to total seconds
    df['end_time'] = (
                df['End time'] - df['file_datetime']).dt.total_seconds()  # convert end time difference to total seconds
    df['annotation'] = df['Call']
    df['high_f'] = df['Parameter 1']
    df['low_f'] = df['Parameter 2']
    df = df.loc[:, ['audio_file', 'annotation', 'high_f', 'low_f', 'start_time',
                    'end_time']]  # subset all rows by certain column name
    return df


# make a subfolder for saving modified logs
subfolder_name = "modified_annotations"
# Create the subfolder if it doesn't exist
subfolder_path = os.path.join(directory_path, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# 'DCPP01A_d01_121106_083945.d100.x.wav' is missing a chunk of data between these bounds:
date1 = datetime(2012, 11, 6, 8, 41, 11)
date2 = datetime(2012, 11, 7, 2, 0, 0)
# Calculate the difference in seconds
seconds_difference = (date2 - date1).total_seconds()

# loop through all annotation files and save them in subfolder "modified_annotations"
for file in all_files:
    data = pd.read_excel(file)

    # DCPP01A_d01_121106_083945.d100.x.wav, subtract number of seconds for accurate time windows
    if any(data['Input file'].str.contains('DCPP01A_d01_121106_083945.d100.x.wav')):

        mask = data['Input file'].str.contains('DCPP01A_d01_121106_083945.d100.x.wav')
        subset_df = modify_annotations(data)

        subset_df.loc[mask, 'start_time'] -= seconds_difference
        subset_df.loc[mask, 'end_time'] -= seconds_difference

    else:
        subset_df = modify_annotations(data)

    filename = os.path.basename(file)
    new_filename = filename.replace('.xls', '_modification.csv')
    # Construct the path to save the modified DataFrame as a CSV file
    save_path = os.path.join(subfolder_path, new_filename)
    # Save the subset DataFrame to the subset folder as a CSV file
    subset_df.to_csv(save_path, index=False)