#coding:utf8
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A multi-layers convolutional network with the following architecture:
  
  The structure of convnet: [conv-relu-pool]xN - [affine-relu]xM -affine- [softmax or SVM]
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), conv_dims=[(32,7)], conv_stride=1, pad = -1,
               pool_height=2, pool_width=2, pool_stride=2,
               hidden_dims=[100], num_classes=10, use_batchnorm=False, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32):
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
    for i in xrange(conv_size): #注意：这里索引值从0开始
        #初始化卷积层的W 和 b
        num_filters = conv_dims[i][0]
        filter_size = conv_dims[i][1]
        
        self.params['W'+str(i)] = weight_scale * np.random.randn(num_filters, channel_size , filter_size, filter_size)
        self.params['b'+str(i)] = np.zeros((num_filters,))
        channel_size = num_filters # 这一轮的 channel_size 是上一轮的 num_filters
        
        # 初始化卷积层的参数
        if pad == -1:
            pad = (filter_size - 1) / 2
        self.conv_params.append( {'stride': conv_stride , 'pad': pad} )
        self.pool_params.append( {'pool_height': pool_height , 'pool_width': pool_width, 'stride': pool_stride} )
        
        # 每一次conv-relu-pool 之后，H和W都发生变化。
        H_new = ((H_new + 2*pad -filter_size)/conv_stride + 1) / pool_height
        W_new = ((W_new + 2*pad -filter_size)/conv_stride + 1) / pool_width
    
    # 计算全连接层输入层的向量的size
    fully_connect_input_dim = channel_size * H_new * W_new
    
    #初始化全连接层的W
    self.params['W'+str(conv_size)] = np.random.randn(fully_connect_input_dim, hidden_dims[0]) \
                                      * np.sqrt(2.0/fully_connect_input_dim)
     
    self.params['b'+str(conv_size)] = np.zeros((hidden_dims[0],))
    for i in xrange(conv_size+1, conv_size + len(hidden_dims)):
        idx = i - conv_size - 1
        self.params['W'+str(i)] = np.random.randn(hidden_dims[idx], hidden_dims[idx+1]) * np.sqrt(2.0/hidden_dims[idx])
        self.params['b'+str(i)] = np.zeros((hidden_dims[idx+1],))
    
    last_idx = conv_size + len(hidden_dims)
    self.params['W'+str(last_idx)] = np.random.randn( hidden_dims[len(hidden_dims)-1], num_classes ) \
                                     * np.sqrt(1.0/hidden_dims[len(hidden_dims)-1])
    
    self.params['b'+str(last_idx)] = np.zeros((num_classes,))
    
    
    self.conv_layers = conv_size
    self.hidden_layers = len(hidden_dims)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    The structure of convnet: [conv-relu-pool]xN - [affine-relu]xM -affine- [softmax or SVM]
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    loss_reg = 0
    caches = []
    out = X
    
    # [conv-relu-pool]xN 前向传播
    for i in xrange(self.conv_layers):
        W, b = self.params['W'+str(i)], self.params['b'+str(i)]
        out, cache = conv_relu_pool_forward(out, W, b, self.conv_params[i], self.pool_params[i])
        caches.append(cache)
        loss_reg += np.sum(W**2)
    
    # [affine-relu]xM -affine 前向传播
    for j in xrange(self.conv_layers, self.conv_layers + self.hidden_layers):
        W,b = self.params['W'+str(j)], self.params['b'+str(j)]
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
    for j in xrange(self.conv_layers + self.hidden_layers-1 , self.conv_layers-1 , -1):
        dout, grads['W'+str(j)], grads['b'+str(j)] = affine_relu_backward(dout, caches[j])
        grads['W'+str(j)] += self.reg * self.params['W'+str(j)]
    
    # [conv-relu-pool]xN 反向传播
    for i in xrange(self.conv_layers-1 , -1 , -1):
        dout, grads['W'+str(i)], grads['b'+str(i)] = conv_relu_pool_backward(dout, caches[i])
        grads['W'+str(i)] += self.reg * self.params['W'+str(i)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

pass
