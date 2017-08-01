#Preprocess XRay images

## Version 0.2
* support cross validation, e.g. 10 folds.
* command line parameters.
* TODO: put the `label_dict` into a text file.

## Initial version 0.1
* Generate meta list
* Split the training and test by 9:1

*Notice: the training program requires `num_test % batch_size = 0`, while this script does not take batch_size into account.
 So we should remove the remainder of the test samples manually*
