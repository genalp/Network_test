import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Preprocessor
import cv2
import LayersConstructor

from datetime import timedelta

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
batch_size = 8

#验证集大小
validation_size = 0.16

early_stopping = None

train_path = 'C:/Users/wzx12/Desktop/Python_test/CNN/dogs_vs_cats/data/train'
test_path = 'C:/Users/wzx12/Desktop/Python_test/CNN/dogs_vs_cats/data/test'

data = Preprocessor.read_train_sets(train_path, img_size, classes, 
	validation_size = validation_size)
test_images, test_ids = Preprocessor.read_test_set(test_path, img_size)
images, cls_true = data.train.images, data.train.cls

print("Size of:")
print("  - Training-set:\t\t{}".format(len(data.train._labels)))
print("  - Test-set:\t\t{}".format(len(test_images)))
print("  - Validation-set:\t{}".format(len(data.valid._labels)))

#定义CNN超参数
#卷积层

filter_size1 = 8
num_filters1 = 96

filter_size2 = 5
num_filters2 = 256

filter_size3 = 3
num_filters3 = 384

filter_size4 = 3
num_filters4 = 384

filter_size5 = 3
num_filters5 = 256

#全连接层
#神经元数量
fc_size = 512
#学习率
learning_rate = 0.0004


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
								use_pooling = True,
                                step = 4
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
								use_pooling = False 
								)

#第四个卷积层
layer_conv4, weights_conv4 = LayersConstructor.new_conv_layer(input = layer_conv3, 
                                num_input_channels = num_filters3, 
                                filter_size = filter_size4, 
                                num_filters = num_filters4, 
                                use_pooling = False 
                                )

#第五个卷积层
layer_conv5, weights_conv5 = LayersConstructor.new_conv_layer(input = layer_conv4, 
                                num_input_channels = num_filters4, 
                                filter_size = filter_size5, 
                                num_filters = num_filters5, 
                                use_pooling = True 
                                )

#扁平层
layer_flat, num_features = LayersConstructor.flatten_layer(layer_conv3)

#全连接层
layer_fc1 = LayersConstructor.new_fc_layer(input = layer_flat, 
										num_inputs = num_features, 
										num_outputs = fc_size, 
										use_relu = True,
                                        dropout = True
										)
layer_fc2 = LayersConstructor.new_fc_layer(input = layer_fc1, 
                                        num_inputs = fc_size, 
                                        num_outputs = fc_size, 
                                        use_relu = True, 
                                        dropout = True
                                        )
layer_fc3 = LayersConstructor.new_fc_layer(input = layer_fc2, 
										num_inputs = fc_size, 
										num_outputs = num_classes, 
										use_relu = False,
                                        dropout = False
										)

#运行Tensorflow图训练CNN模型
#通过softmax来预测类别，与真是类别进行对照
y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc3,
                                                        labels=y_true)
#定义损失函数和优化器，计算精度
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#初始化操作
init_op = tf.global_variables_initializer()

session = tf.Session()
session.run(init_op)
train_batch_size = batch_size

#记录训练和检验准确性
acc_list = []
val_loss_list = []

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
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    patience = 0

    for i in range(total_iterations, total_iterations + num_iterations):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        feed_dict_train = {x: x_batch, y_true: y_true_batch}        
        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)        

        # if i % int(data.train.num_examples/batch_size) == 0: 
        if i % 200 == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            acc, val_acc = print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
            print(msg.format(epoch + 1, acc, val_acc, val_loss))
            print(acc)
            acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            iter_list.append(i+1)

    total_iterations += num_iterations

#模型评估
def print_validation_accuracy(show_example_errors=False, show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.valid.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + batch_size, num_test)
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
        labels = data.valid.labels[i:j, :]
        feed_dict = {x: images, y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

optimize(num_iterations = 14000)
print(acc_list)
# Plot loss over time
plt.plot(iter_list, acc_list, 'r-', label='CNN validation accuracy per iteration', linewidth=1)
plt.title('CNN validation accuracy per iteration')
plt.xlabel('Iteration')
plt.ylabel('CNN validation accuracy')
plt.legend(loc='upper right')
plt.show()

# Plot loss over time
plt.plot(iter_list, val_loss_list, 'r-', label='CNN validation loss per iteration', linewidth=1)
plt.title('CNN validation loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('CNN validation loss')
plt.legend(loc='upper right')
plt.show()  

print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)
plt.axis('off')

session.close()