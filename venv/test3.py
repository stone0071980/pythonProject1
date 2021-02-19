import torch

"""
A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.
一个全连接网络模型，激活函数是ReLU，具有一个隐藏层且没有偏差，经过训练可以使用欧几里得误差根据x来预测y。

This implementation uses PyTorch tensors to manually compute the forward pass,
loss, and backward pass.
该程序实现使用PyTorch张量手动计算前向传播，损失和后向传播。

A PyTorch Tensor is basically the same as a numpy array: it does not know
anything about deep learning or computational graphs or gradients, and is just
a generic n-dimensional array to be used for arbitrary numeric computation.
PyTorch张量基本上与numpy数组相同：它对深度学习，计算图或梯度一无所知，只是用于任意数值计算的通用n维数组。

The biggest difference between a numpy array and a PyTorch Tensor is that
a PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU,
just pass a different value to the `device` argument when constructing the
Tensor.
numpy数组和PyTorch张量之间的最大区别是PyTorch张量可以在CPU或GPU上运行。要在GPU上运行操作，只需在构造Tensor时将不同的值传递给device参数即可。
"""

device = torch.device('cpu') # CPU环境
# device = torch.device('cuda') # Uncomment this to run on GPU GPU环境

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device)  # 输入 (64,1000)
y = torch.randn(N, D_out, device=device)  # 输出 (64,10)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device)  # 输入层-隐藏层 权重 (1000,100)
w2 = torch.randn(H, D_out, device=device)  # 隐藏层-输出层 权重 (100,10)

learning_rate = 1e-6  # 学习率
for t in range(500):
    # Forward pass: compute predicted y 前向传播：计算预测的y
    h = x.mm(w1)  # 点乘 得到隐藏层 (64,100)
    # torch.mm()矩阵相乘
    # torch.mul() 矩阵位相乘
    h_relu = h.clamp(min=0)  # 计算relu激活函数
    # torch.clamp(input, min, max, out=None) → Tensor 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    y_pred = h_relu.mm(w2)  # 点乘 得到输出层 (64,10)

    # Compute and print loss; loss is a scalar标量, and is stored in a PyTorch Tensor
    # of shape (); we can get its value as a Python number with loss.item().
    loss = (y_pred - y).pow(2).sum() # .sum()所有元素的总和 torch.Size([])
    print(t, loss.item()) # pytorch中的.item()用于将一个零维张量转换成浮点数

    # Backprop to compute gradients of w1 and w2 with respect to loss
    # 反向传播的过程（难点），具体过程同上，没有变化
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    # torch.clone()和torch.copy()应该没什么区别
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent  更新权重
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2