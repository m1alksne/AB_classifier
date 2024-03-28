# AB_classifier
A prototype ResNet-18 CNN trained to classify Blue whale A calls and B calls in 30 second spectrogram windows. Here we convert call annotation files (.xls) to hot_clips that contain binary labels for overlapping 30 second windows in wav files. The wav files are not uploaded to Github. The train.py script allows the user to experiment with various data augmentation techniques and hyperparameters using the opensource audio and machine learning package, opensoundscape. 
Start with "modify_annotations.py" script. This allows the user to convert their annotation files into opensoundscape annotation format. 
"make_dataset.py" generates hot_clips for train and validation and test datasets. 
Once your dataset is made, run "train.py" to run the model. 
"predict.py" allows the user to test their model on a new dataset and evaluate predictions using PR-curves and score distribution histograms
"evaluate_model_metrics.py" allows the user to plot average precision per model training epoch 

