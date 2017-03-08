# Healthcare Image Classification

## Repository content

* **data**: raw data and pre-processed data 
* **meta-process**: generating meta-info from raw data
* **data-loader**: loading data for training at runtime 
* **model**: models used for image classification
* **train.py**: main entry for training

## Run steps
* prepare data and run ./preprocess/preprocess.py
* run `python train_new.py` to train 18-layer resnet
