import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle


# Get the location path where the image is stored
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
folder = dir_path +'\\' +image_path
img_sze=256
channel_sze=3
labels = []
predicts = []
classes = ['2061', '6051', '6599', '9633', '9779']
n_classes = len(classes)
i = 0
sess = tf.Session()
saver = tf.train.import_meta_graph('model.meta')
#load the weights
saver.restore(sess, tf.train.latest_checkpoint('./'))
for each_class in classes:   
    index = classes.index(each_class)
    print('Reading Files from {} class'.format(each_class))
    path = os.path.join(folder, each_class, '*g')
    files = glob.glob(path)
    for file1 in files:
        i+=1
        images = []
        image = cv2.imread(file1)
        image = cv2.resize(image, (img_sze, img_sze),0,0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0) 
        label = np.zeros(n_classes)
        label[index] = 1.0
        labels.append(label)
        flbase = os.path.basename(file1)
        #we reshape.
        x_batch = images.reshape(1, img_sze,img_sze,channel_sze)

        # Accessing the  graph
        graph = tf.get_default_graph()
 
        #  y_pred  is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        ## Let's feed the images to the input placeholders
        x= graph.get_tensor_by_name("x:0") 
        y_true = graph.get_tensor_by_name("y_true:0") 
        y_test_images = np.zeros((1, n_classes)) 


        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        predicts.append(result[0])
        # result is of this format [probabiliy_of_rose probability_of_sunflower]
        print(i)

sess.close()
np_labels = np.asarray(labels, np.float32)
predicts_np = np.asarray(predicts, np.float32)

sess = tf.InteractiveSession() 
tensorflow_labels = tf.convert_to_tensor(np_labels, np.float32)
predicts_tf = tf.convert_to_tensor(predicts_np, np.float32)
finalLabel = tf.argmax(tensorflow_labels, 1)
finalPred = tf.argmax(predicts_tf,1)
acc, acc_op = tf.metrics.accuracy(finalLabel,finalPred)
conf = tf.confusion_matrix(finalLabel,finalPred,num_classes=n_classes)
#auc = tf.metrics.auc(finalLabel,finalPred)

# Compute macro-average ROC curve and ROC area
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
#print(predicts_tf.eval())
print("Accuracy: " + str(sess.run([acc, acc_op])[1]*100)+"%")
print("Confusion Matrix: ")
print(sess.run(conf))
#print("Area Under ROC Curve: " + str(sess.run(auc)[1]))
maxLabel = np.argmax(np_labels, axis=1)
maxPredict = np.argmax(predicts_np, axis=1)
print("Classification Report")
print(metrics.classification_report(maxLabel, maxPredict))

#For multi-class ROC curve
y_test_binary = label_binarize(maxLabel, classes=[0, 1, 2, 3, 4])
predictions_binary = label_binarize(maxPredict, classes=[0, 1, 2, 3, 4])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], predictions_binary[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'green', 'blue', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
        label='ROC curve of class {0} (area = {1:0.2f})'
        ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

sess.close()