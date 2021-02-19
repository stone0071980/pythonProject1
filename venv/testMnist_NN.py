import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 下载并加载数据

mnist = input_data.read_data_sets(r'C:\DeepLearning\MnistData', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction

    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

    return result


# 定义Weight变量，输入shape，返回变量的参数。使用了tf.truncted_normal产生随机变量来进行初始化

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)


# 定义biase变量，输入shape，返回变量的一些参数。使用tf.constant常量函数来进行初始化

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)


# 定义卷积操作。tf.nn.conv2d函数是Tensorflow里面的二维的卷积函数，x是图片的所有参数，W是卷积层的权重，然后定义步长strides=[1,1,1,1]值。

# strides[0]和strides[3]的两个1是默认值，意思是不对样本个数和channel进行卷积，中间两个1代表padding是在x方向运动一步，y方向运动一步，

# padding采用的方式实“SAME”就是0填充。

def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]

    # Must have strides[0] = strides[3] =1

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")  # padding="SAME"用零填充边界


# 定义池化操作。为了得到更多的图片信息，卷积时我们选择的是一次一步，也就是strides[1]=strides[2]=1,这样得到的图片尺寸没有变化，

# 而我们希望压缩一下图片也就是参数能少一些从而减少系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。

# pooling有两种，一种是最大值池化，一种是平均值池化，我采用的是最大值池化tf.max_pool()。池化的核函数大小为2*2，因此ksize=[1,2,2,1]，步长为2，

# 因此strides=[1,2,2,1]。

def max_pool_2x2(x):  # 最大池化方案

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# #################处理图片##################################

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32, [None, 784])  # 28*28

ys = tf.placeholder(tf.float32, [None, 10])

# 定义dropout的输入，解决过拟合问题

keep_prob = tf.placeholder(tf.float32)

# 处理xs，把xs的形状变成[-1,28,28,1]

# -1代表先不考虑输入的图片例子多少这个维度。

# 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1。如果是RGB图像，那么channel就是3.

x_image = tf.reshape(xs, [-1, 28, 28, 1])

# print(x_image.shape) #[n_samples, 28,28,1]

# #################处理图片##################################


# # convl layer （卷积层1）##

W_conv1 = weight_variable([5, 5, 1, 32])  # 权重矩阵：kernel 5*5, channel is 1, out size 32(输出32个特征图)

b_conv1 = bias_variable([32])  # 偏置

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32

# （采用RELU修正线性激活单元激活函数）因为采用了SAME的padding方式，输出图片的大小没有变化依然是28*28，只是厚度变厚，因此输出变成28*28*32。

h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32

# # conv2 layer （卷积层2）##

W_conv2 = weight_variable([5, 5, 32, 64])  # kernel 5*5, in size 32, out size 64

b_conv2 = bias_variable([64])  # 偏置要和bias值相同

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64

h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

# # funcl layer （全连接层）##

W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])

# [n_samples,7,7,64]->>[n_samples, 7*7*64]，通过tf.reshape()将h_pool2输出值从一个三维变为一维的数据，-1表示先不考虑输入图片例子维度，将上一个输出结果展平。

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 进行相乘，注意不是卷积了

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过拟合

# func2 layer （最后一层，采用softmax分类器分类）##

W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# #################优化神经网络##################################

# the error between prediction and real data（交叉熵损失函数）

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)  # 梯度下降找最小损失值

# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()

# important step

sess.run(tf.initialize_all_variables())  # 初始化所有变量

# #################优化神经网络##################################


for i in range(1000):  # 训练1000次，每40次检查一下模型的精度

    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

    if i % 40 == 0:
        # print(sess.run(prediction,feed_dict={xs:batch_xs}))

        print(compute_accuracy(mnist.test.images, mnist.test.labels))