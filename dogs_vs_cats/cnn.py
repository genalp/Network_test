import time
import math
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Preprocessor
import cv2
import LayersConstructor

from sklearn.metrics import confusion_matrix
from datetime import timedelta
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

#颜色通道数
num_channels = 3

img_size = 128

#单维度大小
img_size_flat = img_size * img_size * num_channels

img_shape = (img_size, img_size)

#定义标签
classes = ['dogs', 'cats']
num_classes = len(classes)

#需要训练的批大小
batch_size = 14

#验证集大小
validation_size = 0.16

early_stopping = None

train_path = 'C:/Users/wzx12/Desktop/Python_test/CNN/dogs_vs_cats/data/train'
test_path = 'C:/Users/wzx12/Desktop/Python_test/CNN/dogs_vs_cats/data/test'
checkpoint_dir = "C:/Users/wzx12/Desktop/Python_test/CNN/dogs_vs_cats/model"

data = Preprocessor.read_train_sets(train_path, img_size, classes, 
	validation_size = validation_size)
test_images, test_ids = Preprocessor.read_test_set(test_path, img_size)

def plot_images(images, cls_true, cls_pred = None):
	if len(images) == 0:
		print("no images to show")
		return
	else:
		random_indices = random.sample(range(len(images)), min(len(images), 9))
		images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
	for i, ax in enumerate(axes.flat):
		#plot image
		ax.imshow(images[i].reshape(img_size, img_size, num_channels))
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
		ax.set_xlabel(xlabel)
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

images, cls_true = data.train.images, data.train.cls

# plot_images(images = images, cls_true = cls_true)

print("Size of:")
print("  - Training-set:\t\t{}".format(len(data.train._labels)))
print("  - Test-set:\t\t{}".format(len(test_images)))
print("  - Validation-set:\t{}".format(len(data.valid._labels)))


#定义CNN超参数
#卷积层
filter_size1 = 3
num_filters1 = 32

filter_size2 = 3
num_filters2 = 32

filter_size3 = 3
num_filters3 = 64

#全连接层
#神经元数量
fc_size = 128
#学习率
learning_rate = 1e-4


#构造CNN层


#准备TensorFlow图
x = tf.placeholder(tf.float32, shape = [None, img_size_flat], name = 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')
y_true_cls = tf.argmax(y_true, axis = 1)

#创建CNN模型
#第一个卷积层
layer_conv1, weights_conv1 = LayersConstructor.new_conv_layer(input = x_image, 
								num_input_channels = num_channels, 
								filter_size = filter_size1, 
								num_filters = num_filters1,
								use_pooling = True
								)

#第二个卷积层
layer_conv2, weights_conv2 = LayersConstructor.new_conv_layer(input = layer_conv1, 
								num_input_channels = num_filters1, 
								filter_size = filter_size2, 
								num_filters = num_filters2, 
								use_pooling = True 
								)

#第三个卷积层
layer_conv3, weights_conv3 = LayersConstructor.new_conv_layer(input = layer_conv2, 
								num_input_channels = num_filters2, 
								filter_size = filter_size3, 
								num_filters = num_filters3, 
								use_pooling = True 
								)

#扁平层
layer_flat, num_features = LayersConstructor.flatten_layer(layer_conv3)

#全连接层
layer_fc1 = LayersConstructor.new_fc_layer(input = layer_flat, 
										num_inputs = num_features, 
										num_outputs = fc_size, 
										use_relu = True 
										)
layer_fc2 = LayersConstructor.new_fc_layer(input = layer_fc1, 
										num_inputs = fc_size, 
										num_outputs = num_classes, 
										use_relu = True 
										)

#运行Tensorflow图训练CNN模型
#通过softmax来预测类别，与真是类别进行对照
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis = 1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer_fc2, 
															labels = y_true)
#定义损失函数和优化器，计算精度
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#初始化操作
init_op = tf.global_variables_initializer()

session = tf.Session()
session.run(init_op)
train_batch_size = batch_size

#记录训练和检验准确性
acc_list = []
val_acc_list = []

#记录迭代
total_iterations = 0
iter_list = []

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    return acc, val_acc

#开始正式训练
def optimize(num_iterations):
    global total_iterations    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations, total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}        
        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)        

        # Print status at end of each epoch (defined as full pass through training Preprocessor).
        if i % int(data.train.num_examples / batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))           
            acc, val_acc = print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            acc_list.append(acc)
            val_acc_list.append(val_acc)
            iter_list.append(epoch+1)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

#模型评估
def plot_example_errors(cls_pred, correct):
	incorrect = (correct == False)
	images = data.valid.images[incorrect]
	cls_pred = cls_pred[incorrect]
	plot_images(images = images[0:9], cls_true = cls_true[0:9], cls_pred = cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.valid.cls
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    p, r, f, s = precision_recall_fscore_support(cls_true, cls_pred, average='weighted')
    print('Precision:', p)
    print('Recall:', r)
    print('F1-score:', f)

    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

def print_validation_accuracy(show_example_errors=False, show_confusion_matrix=False):
    num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)

        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images, y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred) 

optimize(num_iterations = 1000)
print(acc_list)
# Plot loss over time
plt.plot(iter_list, acc_list, 'r--', label='CNN training accuracy per iteration', linewidth=4)
plt.title('CNN training accuracy per iteration')
plt.xlabel('Iteration')
plt.ylabel('CNN training accuracy')
plt.legend(loc='upper right')
plt.show()

# Plot loss over time
plt.plot(iter_list, val_acc_list, 'r--', label='CNN validation accuracy per iteration', linewidth=4)
plt.title('CNN validation accuracy per iteration')
plt.xlabel('Iteration')
plt.ylabel('CNN validation accuracy')
plt.legend(loc='upper right')
plt.show()  

print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)
plt.axis('off')