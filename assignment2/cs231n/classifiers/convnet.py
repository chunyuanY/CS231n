#coding:utf8
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

#本文件是我自己根据fc_net.py和cnn.py改写的卷积神经网络，网络结构如下：
# [conv-bn?-relu-pool]xN - [affine-bn?-relu-dropout?]xM -affine- [softmax or SVM]

# 1.在卷积层和全连接层都使用了batch normalization. 
#   使用batch normal最大的好处是，即使参数设置的不好，最终也能得到一个很不错的结果，而且收敛速度也会加快。
#   因此，这里默认开启batch normal.

# 2.这里只在全连接层使用dropout，因为卷积层有pool就已经足够减轻过拟合了，而且pool还能实现平移不变性。
#   这里默认不开启dropout, 需要dropout时，设置 0 < dropout < 1即可。

# TODO: 有时间可以尝试一下其他类型的网络结构，比如说： 
# [conv-bn-relu-conv-bn-relu-pool]xN - [affine-bn-relu]xM -affine- [softmax or SVM]
# 如果硬件条件允许的话，也可以尝试一下一些著名的网络结构：AlexNet, VGGNet, ZFNet, GoogLeNet等。

# TODO: 可以尝试用Leaky ReLU, Parametric ReLU, or MaxOut 代替 ReLU。

# TODO: 可以尝试data argmentation来扩充数据集。可用的方式如下：
# 1）水平翻转（Horizontal flips）
# 2）随机剪裁（Random crops/scales）
# 3）色彩抖动（Color jitter）
# 4）平移、旋转、拉伸、切变、光学畸变等等。
# 可参考：http://www.jianshu.com/p/9c4396653324

class ConvNet(object):
  """
  A multi-layers convolutional network with the following architecture:
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), conv_dims=[(32,7)], conv_stride=1, pad = -1,
               pool_height=2, pool_width=2, pool_stride=2,
               hidden_dims=[100], num_classes=10, 
               use_batchnorm_in_conv=False, use_batchnorm_in_fc = False, dropout=0,
               weight_scale=1e-3, reg=0.0, dtype=np.float32 , seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - conv_dims: A list of tuples. 
            tuple[i][0] giving the num_filters of ith convolutional layer.
            tuple[i][1] giving the filter_size of ith convolutional layer.
    
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dims: A list of integers giving the size of each hidden fully connected layer.
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm_in_conv = use_batchnorm_in_conv
    self.use_batchnorm_in_fc = use_batchnorm_in_fc
    self.use_dropout = dropout > 0
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # pass conv_param to the forward pass for the convolutional layer
    self.conv_params = []
    # pass pool_param to the forward pass for the max-pooling layer
    self.pool_params = []
    
    
    H_new = input_dim[1]
    W_new = input_dim[2]
    conv_size = len(conv_dims)
    channel_size = 3
    for i in range(conv_size): #注意：这里索引值从0开始
        #初始化卷积层的W 和 b
        num_filters = conv_dims[i][0]
        filter_size = conv_dims[i][1]
        
         # 初始化conv layer的配置信息
        if pad == -1:
            assert (filter_size - 1) % 2 == 0, "filter size is not good!"
            pad = int( (filter_size - 1) / 2 )
        
        self.conv_params.append( {'stride': conv_stride , 'pad': pad} )
        self.pool_params.append( {'pool_height': pool_height , 'pool_width': pool_width, 'stride': pool_stride} )
        
        # 初始化conv layer的权重
        self.params['W'+str(i)] = weight_scale * np.random.randn(num_filters, channel_size , filter_size, filter_size)
        self.params['b'+str(i)] = np.zeros((num_filters,))
        channel_size = num_filters # 这一轮的 channel_size 是上一轮的 num_filters
        
        # 初始化卷积层batch normalization的参数
        if self.use_batchnorm_in_conv:
            self.params['gamma'+str(i)] = np.ones(channel_size)
            self.params['beta'+str(i)] = np.zeros(channel_size)
        
        # 每一次conv-relu-pool 之后，H和W都发生变化。
        assert (H_new + 2*pad -filter_size) % conv_stride == 0, "convolution stride is not good!"
        assert (W_new + 2*pad -filter_size) % conv_stride == 0, "convolution stride is not good!"
        assert ((H_new + 2*pad -filter_size)/conv_stride + 1) % pool_height == 0, "pool height is not good!"
        assert ((W_new + 2*pad -filter_size)/conv_stride + 1) % pool_width == 0, "pool width is not good!"
        H_new = int( ((H_new + 2*pad -filter_size)/conv_stride + 1) / pool_height )
        W_new = int( ((W_new + 2*pad -filter_size)/conv_stride + 1) / pool_width )
    
    
    # 计算全连接层输入层的向量的size
    fc_input_dim = channel_size * H_new * W_new
    #初始化全连接层的W
    self.params['W'+str(conv_size)] = np.random.randn(fc_input_dim, hidden_dims[0]) * np.sqrt(2.0/fc_input_dim)
    self.params['b'+str(conv_size)] = np.zeros(hidden_dims[0])
    
    # 初始化全连接层batch normalization的参数
    if self.use_batchnorm_in_fc:
        self.params['gamma'+str(conv_size)] = np.ones(hidden_dims[0])
        self.params['beta'+str(conv_size)] = np.zeros(hidden_dims[0])
    
    for i in range(conv_size+1, conv_size + len(hidden_dims)):
        idx = i - conv_size - 1  # idx起始值是0
        
        # 初始化fully-connected layer的W和b
        self.params['W'+str(i)] = np.random.randn(hidden_dims[idx], hidden_dims[idx+1]) * np.sqrt(2.0/hidden_dims[idx])
        self.params['b'+str(i)] = np.zeros((hidden_dims[idx+1],))
        
        # 初始化全连接层batch normalization的参数
        if self.use_batchnorm_in_fc:
            self.params['gamma'+str(i)] = np.ones(hidden_dims[idx])
            self.params['beta'+str(i)] = np.zeros(hidden_dims[idx])
            
    last_idx = conv_size + len(hidden_dims)
    self.params['W'+str(last_idx)] = np.random.randn( hidden_dims[len(hidden_dims)-1], num_classes ) \
                                     * np.sqrt(1.0/hidden_dims[len(hidden_dims)-1])
    self.params['b'+str(last_idx)] = np.zeros((num_classes,))
    
    
    self.conv_layers = conv_size
    self.hidden_layers = len(hidden_dims)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # 初始化batch normal的配置信息
    self.bn_params = [{} for _ in range(self.conv_layers + self.hidden_layers)]
    if self.use_batchnorm_in_conv:
        for i in range(self.conv_layers):
          self.bn_params[i] = {'mode': 'train'}
    if self.use_batchnorm_in_fc:
        for j in range(self.conv_layers, self.conv_layers+self.hidden_layers):
          self.bn_params[j] = {'mode': 'train'}
    
    # 初始化dropout的配置信息
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    The structure of convnet: [conv-relu-pool]xN - [affine-relu]xM -affine- [softmax or SVM]
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    mode = 'test' if y is None else 'train'
    
    #重新设置batch normal的mode
    if self.use_batchnorm_in_conv or self.use_batchnorm_in_fc:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    
    #重新设置dropout的mode
    if self.use_dropout:
      self.dropout_param['mode'] = mode  
    

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    loss_reg = 0
    caches = []
    out = X
    
    # [conv-bn?-relu-pool]xN 前向传播
    for i in range(self.conv_layers):
        W, b = self.params['W'+str(i)], self.params['b'+str(i)]
        if self.use_batchnorm_in_conv:
            out, cache = conv_bn_relu_pool_forward(out, W, b, 
                                                   self.conv_params[i], 
                                                   self.pool_params[i],
                                                   self.params['gamma'+str(i)],
                                                   self.params['beta'+str(i)],
                                                   self.bn_params[i]
                                                  )
        else:
            out, cache = conv_relu_pool_forward(out, W, b, self.conv_params[i], self.pool_params[i])
        
        caches.append(cache)
        loss_reg += np.sum(W**2)
    
   
    # [affine-bn?-relu-dropout?]xM -affine 前向传播
    for j in range(self.conv_layers, self.conv_layers + self.hidden_layers):
        W,b = self.params['W'+str(j)], self.params['b'+str(j)]
        
        if self.use_dropout and self.use_batchnorm_in_fc:
            out, cache = affine_bn_relu_dropout_forward(out,W,b,
                                                self.params['gamma'+str(j)],
                                                self.params['beta'+str(j)],
                                                self.bn_params[j],
                                                self.dropout_param
                                               )
        elif self.use_batchnorm_in_fc:
            out, cache = affine_bn_relu_forward(out,W,b,
                                                self.params['gamma'+str(j)],
                                                self.params['beta'+str(j)],
                                                self.bn_params[j]
                                               )
        else:
            out, cache = affine_relu_forward(out,W,b)
        
        caches.append(cache)
        loss_reg += np.sum(W**2)
    
    last_idx = self.conv_layers + self.hidden_layers
    W,b = self.params['W'+str(last_idx)], self.params['b'+str(last_idx)]
    scores, cache = affine_forward(out, W, b)
    caches.append(cache)
    loss_reg += np.sum(W**2)
    
    loss_reg *= self.reg
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # - [softmax or SVM] 前向传播
    loss_data, dout = softmax_loss(scores, y)
    
    loss = loss_data + loss_reg
    
    # -affine 反向传播
    last_idx = self.conv_layers + self.hidden_layers
    dout, grads['W'+str(last_idx)], grads['b'+str(last_idx)] = affine_backward(dout, caches[last_idx])
    grads['W'+str(last_idx)] += self.reg * self.params['W'+str(last_idx)]
        
    # [affine-relu]xM 反向传播
    for j in range(self.conv_layers + self.hidden_layers-1 , self.conv_layers-1 , -1):
        if self.use_dropout and self.use_batchnorm_in_fc:
            dout, grads['W'+str(j)], grads['b'+str(j)],\
                  grads['gamma'+str(j)], grads['beta'+str(j)] =\
                                            affine_bn_relu_dropout_backward(dout, caches[j])
        elif self.use_batchnorm_in_fc:
            dout, grads['W'+str(j)], grads['b'+str(j)],\
                  grads['gamma'+str(j)], grads['beta'+str(j)] =\
                                            affine_bn_relu_backward(dout, caches[j])
        else:
            dout, grads['W'+str(j)], grads['b'+str(j)] = affine_relu_backward(dout, caches[j])
        
        grads['W'+str(j)] += self.reg * self.params['W'+str(j)]
    
    # [conv-relu-pool]xN 反向传播
    for i in range(self.conv_layers-1 , -1 , -1):
        if self.use_batchnorm_in_conv:
            dout, grads['W'+str(i)], grads['b'+str(i)],\
                  grads['gamma'+str(i)], grads['beta'+str(i)] =\
                                           conv_bn_relu_pool_backward(dout, caches[i])
        else:
            dout, grads['W'+str(i)], grads['b'+str(i)] = conv_relu_pool_backward(dout, caches[i])
        
        grads['W'+str(i)] += self.reg * self.params['W'+str(i)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

pass
