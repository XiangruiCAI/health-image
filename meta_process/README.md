# Preprocessing scripts to generate meta data for the healthcare images
## run
These scripts suppose your data are put at `path_to_repo/data/raw` and the
records.csv is put at `path_to_repo/data`.

First `python preprocess.py` then `python combine.py`

*PLEASE USE meta_final.csv and meta_full_final.csv AS THE INPUT FOR THE FOLLOW UP WORK!*

## preprocessing.py
* input: records.csv and the iamge folder.
* output: 
    * meta.csv: a file containing path, label  with space as the delimiter(for training).
    * meta_full.csv: a file containing path, label, IID, gender, with space as the delimiter(for intepretation).
    * preprocessing.log: a log file with all information
    * nopa.log: a log file containing the list of images which have no PA-like fields
* dependency: `pip install pydicom`


## combine.py
Add iamges that are found manually into the final list
* input: meta_full.csv and manually.csv
* output: meta_final.csv and meta_full_final.csv, both are the corresponding files to the output of preprocessing.py

