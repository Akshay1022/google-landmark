As mentioned in the report, there are two types of models in this project.

1) Binary classification (between 2 most frequent landmarks)
2) Multi-classification (between 5 most frequent landmarks)

The code is perfectly generic and it can work for any number of classes. We just have to perform the following steps.
1) Download the training dataset images from the url and load them in local directories. Create separate folder for each class. (This step can be easily performed by the provided script "download-images.py"
2) Change the class names in createModel.py at line 19.
3) Change the location of the training dataset in createModel.py at line 27.
4) Change the location where you want to save the model in createModel.py at line 200.

Pre-requisites:
---------------

-Python installed with environment variable setup
-Tensor Flow
Folder structure should be created
Download train and test csv dataset from Kaggle.

How to download images from urls in the csv:
--------------------------------------------

python download-images.py <dataset_file.csv> <output_dir>

Here give the path of the csv of the dataset which contains the imageid, image url and image class.
output_dir is the directory of the output.
Now this code will extract the url from csv and start downloading them to system.
We have created multiple folders in output directory and downloading the images to different folders according to 
the class they belong to.
Run the same command on the test.csv to obtain test images downloaded in different folders according to class.

Commands:
--------

To create model from training data:
-----------------------------------
-Change the location of the training dataset in createModel.py at line 27.
-Change the location where you want to save the model in createModel.py at line 200.

python createModel.py

the file has variable train_path which takes the directory of input images. This directory is the same as <output_dir>
above.
classes variable has the names of the folders of different classes of images stored in the directory. The code appends this
class name with directory and takes images from each folder. Here each folder represents each class.
This code outputs epochs having training and validation percentages. It creates a meta file and stores on system.
saver.session uses a folder to save session data. 


Once the above the command is executed, the epochs start printing in the console and accuracies for each epoch are also printed. The model has been saved to given
location after each epoch and hence the script can be stopped at any iteration once we get the good accuracy.

To predict test data:
--------------------

python test.py <location_of_testing_images>

for example: python test.py binary-dataset\testing-data

Give the input directory of the test images. The code loops through all the sub folders and outputs accuracy, precision and other evaluation metrics

#tf.train.import_meta_graph('output.meta') in the code uses the meta file generated during train.py. Give the location of
the file here.
