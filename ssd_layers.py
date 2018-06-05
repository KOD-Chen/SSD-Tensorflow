# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/3$ 11:28$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : ssd_layers$.py
# Description :SSD的build_net()中各种网络层的定义.
# --------------------------------------

import tensorflow as tf

# conv2d卷积层:步长=1
def conv2d(x,filters,kernel_size,stride=1,padding='same',
           dilation_rate=1,activation=tf.nn.relu,scope='conv2d'):
    kernel_sizes = [kernel_size] * 2 # --> [kernel_size,kernel_size]
    strides = [stride] * 2 # --> [stride,stride]
    dilation_rate = [dilation_rate] * 2 # 膨胀率-->[dilation_rate,dilation_rate]
    return tf.layers.conv2d(inputs=x,filters=filters,kernel_size=kernel_sizes,
                            strides=strides,dilation_rate=dilation_rate,padding=padding,
                            name=scope,activation=activation)

# max_pool2d最大池化层
def max_pool2d(x, pool_size, stride=None, scope='max_pool2d'):
    pool_sizes = [pool_size] * 2
    if stride==None:
        strides = [pool_size] * 2
    else:
        strides = [stride] * 2
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_sizes,strides=strides,name=scope,padding='same')

# pad2d零填充：针对步长为1的conv2d层
def pad2d(x,pad):
    return tf.pad(x,paddings=[[0,0],[pad,pad],[pad,pad],[0,0]])

# dropout
def dropout(x,rate=0.5,is_training=True):
    return tf.layers.dropout(inputs=x,rate=rate,training=is_training)

# l2norm：Conv4_3层将作为用于检测的第一个特征图,该层比较靠前，其norm较大，
# 所以在其后面增加了一个L2 Normalization层，以保证和后面的检测层差异不是很大.
# 这个和Batch Normalization层不太一样:其仅仅是对每个像素点在channle维度做归一化，归一化后一般设置一个可训练的放缩变量gamma.
# 而Batch Normalization层是在[batch_size, width, height]三个维度上做归一化。
def l2norm(x,scale,trainable=True,scope='L2Normalization'):
    n_channels = x.get_shape().as_list()[-1] # 通道数
    l2_norm = tf.nn.l2_normalize(x,axis=[3],epsilon=1e-12) # 只对每个像素点在channels上做归一化
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
    return l2_norm * gamma

# multibox layer:
# 从由Conv4_3，Conv7，Conv8_2，Conv9_2，Conv10_2，Conv11_2特征图经过卷积
# 得到的最后detection layer得获取边界框的类别classes、位置location的预测值。
def ssd_multibox_layer(x,num_classes,sizes,ratios,normalization=-1,scope='multibox'):
    pre_shape = x.get_shape().as_list()[1:-1] # 去除第一个和最后一个得到shape
    pre_shape = [-1] + pre_shape
    with tf.variable_scope(scope):
        # l2 norm
        if normalization > 0:
            x = l2norm(x,normalization)
            print(x)

        # anchors数量
        n_anchors = len(sizes) + len(ratios)
        # locations位置预测值
        loc_pred = conv2d(x,filters=n_anchors*4,kernel_size=3,activation=None,scope='conv_loc') # 一个anchor用4个量表示位置、大小
        loc_pred = tf.reshape(loc_pred,pre_shape + [n_anchors,4]) # [anchor数量，每个anchor的locations信息]
        # class类别预测值
        cls_pred = conv2d(x,filters=n_anchors*num_classes,kernel_size=3,activation=None,scope='conv_cls')
        cls_pred = tf.reshape(cls_pred,pre_shape + [n_anchors,num_classes]) # [anchor数量，每个anchor的class信息]

        return cls_pred,loc_pred