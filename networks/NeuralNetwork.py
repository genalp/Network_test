import numpy
import scipy.special
import scipy.ndimage
import matplotlib.pyplot

class NeuralNetwork():

	#初始化神经网络
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#各层节点数
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		#学习率
		self.lr = learningrate

		#权重
		# w11 w21
		# w12 w22 etc
		# self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
		# self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), 
			(self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), 
			(self.onodes, self.hnodes))

		#激活函数
		self.activation_function = lambda x: scipy.special.expit(x)

		pass

	#训练神经网络
	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin = 2).T
		targets = numpy.array(targets_list, ndmin = 2).T

		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		#输出层误差
		output_errors = targets - final_outputs
		#隐藏层误差
		hidden_errors = numpy.dot(self.who.T, output_errors)

		#更新权重
		self.who += self.lr * numpy.dot((output_errors * final_outputs * 
			(1.0 * final_outputs)), numpy.transpose(hidden_outputs))
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * 
			(1.0 - hidden_outputs)), numpy.transpose(inputs))
		pass

	#查询神经网络
	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin = 2).T

		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
		pass

#输入层，隐藏层，输出层节点数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#学习率
learning_rate = 0.1


#创建神经网络
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#载入数据集
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#训练神经网络
eqochs = 1

for e in range(eqochs):
	for record in training_data_list:
		#整理矩阵
		all_values = record.split(',')
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

		#设置目标矩阵
		targets = numpy.zeros(output_nodes) + 0.01
		targets[int(all_values[0])] = 0.99
		n.train(inputs, targets)

		inputs_minus10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 
			-8, cval = 0.01, reshape = False)
		inputs = inputs_minus10.reshape(-1)
		n.train(inputs, targets)
		pass
	pass

#载入测试集
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
	all_values = record.split(',')
	correct_label = int(all_values[0])

	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	outputs = n.query(inputs)
	label = numpy.argmax(outputs)
	if label == correct_label:
		scorecard.append(1)
	else:
		scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)

# #显示图像
# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None')
# matplotlib.pyplot.show()

# matplotlib.pyplot.imshow(output, cmap = 'Greys', interpolation = 'None')
# matplotlib.pyplot.show()