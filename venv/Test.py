import pandas as pd
import numpy as np
import torch
import torchvision

import matplotlib as mpl

import gym

# require 'torch'

# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(190):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.reset()
# env.close()



"""
A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x using Euclidean error.
一个全连接网络模型，激活函数是ReLU，具有一个隐藏层且没有偏差，经过训练可以使用欧几里得误差根据x来预测y。

This implementation uses numpy to manually compute the forward pass, loss, and
backward pass.
该程序实现了使用numpy手动计算前向传播，损失和后向传播。

A numpy array is a generic n-dimensional array; it does not know anything about
deep learning or gradients or computational graphs, and is just a way to perform
generic numeric computations.
numpy数组是通用的n维数组；它对深度学习，梯度或计算图一无所知，只是执行通用数值计算的一种方法。
"""

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)  # 输入 (64,1000)
y = np.random.randn(N, D_out)  # 输出 (64,10)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)  # 输入层-隐藏层 权重 (1000,100)
w2 = np.random.randn(H, D_out)  # 隐藏层-输出层 权重 (100,10)

learning_rate = 1e-6  # 学习率
for t in range(500):
    # Forward pass: compute predicted y 前向传播：计算预测的y
    h = x.dot(w1)  # 点乘 得到隐藏层 (64,100)
    h_relu = np.maximum(h, 0)  # 计算relu激活函数
    # np.maximum(X, Y, out=None) X和Y相比取最大值
    # np.max(a, axis=None, out=None, keepdims=False) 求序列的最值, axis：默认为列向（也即 axis=0），axis = 1 时为行方向的最值
    y_pred = h_relu.dot(w2)  # 点乘 得到输出层 (64,10)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()  # .sum()所有元素的总和
    print(t, loss)  # 目的就是使Loss越来越小

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # 反向传播的过程(难点)，详见下图推导过程
    grad_y_pred = 2.0 * (y_pred - y)  # (64,10)
    grad_w2 = h_relu.T.dot(grad_y_pred)  # (64,100)^T dot (64,10) = (100,10)

    grad_h_relu = grad_y_pred.dot(w2.T)  # (64,100)
    grad_h = grad_h_relu.copy()  # 深拷贝 (64,100)
    grad_h[h < 0] = 0  # Relu反向传播处理过程
    grad_w1 = x.T.dot(grad_h)  # (1000,100)

    # Update weights 更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2