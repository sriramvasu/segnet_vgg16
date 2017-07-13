1) Run the function make_labelimg_fromtxt() in script.py. It will create label image files from text files. It will also label the pixels with no label was white, and create a new folder called new_labels in each of the 3 data folders

2) Run the function train_test_split in script.py. It will create a train val test split in lej folder. Ensure the input to lej folder is given correctly to this function

3) Now run in the train_segnet function in segnet_train_top.py, change the paths to data and label input folder. This new segnet trainer does not optimize the loss function where the label does not exist in the data. 

4) Try running segnet_train_top.py. It should start training.
