#Preprocess XRay images

* Generate meta list
* Split the training and test by 9:1

*Notice: the training program requires `num_test % batch_size = 0`, while this script does not take batch_size into account.
 So we should remove the remainder of the test samples manually*
