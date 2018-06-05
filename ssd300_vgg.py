# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/3$ 15:28$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : ssd300_vgg$.py
# Description :SSD300的网络结构(输入图片为300)、网络输出解码并筛选得分大于阈值的预测边界框box.
# --------------------------------------

from collections import namedtuple
import numpy as np
import tensorflow as tf

from SSD.ssd_layers import conv2d,max_pool2d,l2norm,dropout,pad2d,ssd_multibox_layer
from SSD.ssd_anchors import ssd_anchors_all_layers

# SSD参数
SSDParams = namedtuple('SSDParameters', ['img_shape',  # 输入图片大小: 300x300
                                         'num_classes',  # 类别个数: 20+1（20类+1背景）
                                         'no_annotation_label',
                                         'feature_layers', # 最后detection layer的特征图名字列表
                                         'feature_shapes', # 最后detection layer的特征图size尺寸列表
                                         'anchor_size_bounds', # the down and upper bounds of anchor sizes
                                         'anchor_sizes',   # 最后detection layer的anchor size尺寸列表list
                                         'anchor_ratios',  # 最后detection layer的anchor的长宽比列表list
                                         'anchor_steps',   # list of cell size (pixel size) of layer for detection
                                         'anchor_offset',  # 每个anchor的中心点坐标相对cell左上角的偏移量
                                         'normalizations', # list of normalizations of layer for detection
                                         'prior_scaling'   #
                                         ])

class SSD(object):
    # 构造函数
    def __init__(self,is_training=True):
        self.is_training = is_training
        self.threshold = 0.5 # class score类别分数阈值
        self.ssd_params = SSDParams(img_shape=(300,300),
                                    num_classes=21,
                                    no_annotation_label=21,
                                    feature_layers=['block4','block7','block8','block9','block10','block11'],
                                    feature_shapes=[(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)],
                                    anchor_size_bounds=[0.15, 0.90],  # diff from the original paper
                                    anchor_sizes=[(21.,45.),(45.,99.),(99.,153.),
                                                  (153.,207.),(207.,261.),(261.,315.)],
                                    anchor_ratios=[[2, .5],[2, .5, 3, 1. / 3],[2, .5, 3, 1. / 3],
                                                   [2, .5, 3, 1. / 3],[2, .5],[2, .5]],
                                    anchor_steps=[8, 16, 32, 64, 100, 300],
                                    anchor_offset=0.5,
                                    normalizations=[20, -1, -1, -1, -1, -1],
                                    prior_scaling=[0.1, 0.1, 0.2, 0.2]
                                    )

        predictions,locations = self._built_net() # 【1】SSD300的网络结构(输入图片为300)
        # self._update_feature_shapes_from_net()
        classes,scores,bboxes = self._bboxes_select(predictions,locations) # 【2、3】解码网络输出，并筛选边界框
        self._classes = classes # 类别
        self._scores = scores # 得分(概率)
        self._bboxes = bboxes # 预测边界框的位置和大小

    ##################################【1】SSD300的网络结构(输入图片为300)################################################
    def _built_net(self):
        self.end_points = {}  # 记录detection layers输出
        # 输入图片的占位节点（固定大小的占位）
        self._images = tf.placeholder(tf.float32,
                                      shape=[None,self.ssd_params.img_shape[0],self.ssd_params.img_shape[1],3])

        with tf.variable_scope('ssd_300_vgg'): # 注意："ssd_300_vgg"不能修改，否则导入的模型会找不到
            # (1)原来经典的vgg layers
            # block 1
            net = conv2d(self._images, filters=64, kernel_size=3, scope='conv1_1')
            net = conv2d(net, 64, 3, scope='conv1_2')
            self.end_points['block1'] = net
            net = max_pool2d(net, pool_size=2, scope='pool1')
            # block 2
            net = conv2d(net, 128, 3, scope='conv2_1')
            net = conv2d(net, 128, 3, scope='conv2_2')
            self.end_points['block2'] = net
            net = max_pool2d(net, 2, scope='pool2')
            # block 3
            net = conv2d(net, 256, 3, scope="conv3_1")
            net = conv2d(net, 256, 3, scope="conv3_2")
            net = conv2d(net, 256, 3, scope="conv3_3")
            self.end_points["block3"] = net
            net = max_pool2d(net, 2, scope="pool3")
            # block 4
            net = conv2d(net, 512, 3, scope="conv4_1")
            net = conv2d(net, 512, 3, scope="conv4_2")
            net = conv2d(net, 512, 3, scope="conv4_3")
            self.end_points["block4"] = net
            net = max_pool2d(net, 2, scope="pool4")
            # block 5
            net = conv2d(net, 512, 3, scope="conv5_1")
            net = conv2d(net, 512, 3, scope="conv5_2")
            net = conv2d(net, 512, 3, scope="conv5_3")
            self.end_points["block5"] = net
            print(net)
            net = max_pool2d(net, pool_size=3, stride=1, scope="pool5")  # 'pool核'大小为3*3，步长为1
            print(net)

            # (2)后添加的SSD layers
            # block 6:使用空洞卷积(带膨胀系数的dilate conv)
            net = conv2d(net, filters=1024, kernel_size=3, dilation_rate=6, scope='conv6')
            self.end_points['block6'] = net
            # net = dropout(net, is_training=self.is_training)
            # block 7
            net = conv2d(net, 1024, 1, scope='conv7')
            self.end_points['block7'] = net
            # block 8
            net = conv2d(net, 256, 1, scope='conv8_1x1')
            net = conv2d(pad2d(net,1), 512, 3, stride=2, scope='conv8_3x3', padding='valid')
            self.end_points['block8'] = net
            # block 9
            net = conv2d(net, 128, 1, scope="conv9_1x1")
            net = conv2d(pad2d(net, 1), 256, 3, stride=2, scope="conv9_3x3", padding="valid")
            self.end_points["block9"] = net
            # block 10
            net = conv2d(net, 128, 1, scope="conv10_1x1")
            net = conv2d(net, 256, 3, scope="conv10_3x3", padding="valid")
            self.end_points["block10"] = net
            # block 11
            net = conv2d(net, 128, 1, scope="conv11_1x1")
            net = conv2d(net, 256, 3, scope="conv11_3x3", padding="valid")
            self.end_points["block11"] = net

            # class和location的预测值
            predictions = []
            locations = []
            for i, layer in enumerate(self.ssd_params.feature_layers):
                # layer=self.ssd_params.feature_layers=['block4','block7','block8','block9','block10','block11']
                cls, loc = ssd_multibox_layer(self.end_points[layer], self.ssd_params.num_classes,
                                              self.ssd_params.anchor_sizes[i],
                                              self.ssd_params.anchor_ratios[i],
                                              self.ssd_params.normalizations[i],
                                              scope=layer + '_box')
                predictions.append(tf.nn.softmax(cls))  # 解码class得分：用softmax函数
                locations.append(loc)  # 解码边界框位置xywh

            return predictions, locations

    # 从prediction layers中获得特征图的shapes
    def _update_feature_shape_from_net(self,predictions):
        new_feature_shapes = []
        for l in predictions:
            new_feature_shapes.append(l.get_shape().as_list()[1:])
        self.ssd_params._replace(feature_shapes=new_feature_shapes )

    # 获取SSD的anchor
    def anchors(self):
        return ssd_anchors_all_layers(self.ssd_params.img_shape,
                                      self.ssd_params.feature_shapes,
                                      self.ssd_params.anchor_sizes,
                                      self.ssd_params.anchor_ratios,
                                      self.ssd_params.anchor_steps,
                                      self.ssd_params.anchor_offset,
                                      np.float32)
    ####################################################################################################################

    ###################################【2】解码网络输出，得到边界框位置和大小bbox_location################################
    def _bboxes_decode_layer(self,feature_locations,anchor_bboxes,prior_scaling): # prior_scaling:先验尺寸
        """
        Decode the feature location of one layer
        params:
          feature_locations: 5D Tensor, [batch_size, size, size, n_anchors, 4]
          anchor_bboxes: list of Tensors(y, x, w, h)
                         shape: [size,size,1], [size,size,1], [n_anchors], [n_anchors]
          prior_scaling: list of 4 floats
        """
        y_a,x_a,h_a,w_a = anchor_bboxes
        print(y_a)
        # 解码：由anchor计算真实的cx/cy/w/h
        cx = feature_locations[:,:,:,:,0] * w_a * prior_scaling[0] + x_a #########################
        cy = feature_locations[:,:,:,:,1] * h_a * prior_scaling[1] + y_a
        w = w_a * tf.exp(feature_locations[:,:,:,:,2] * prior_scaling[2])
        h = h_a * tf.exp(feature_locations[:,:,:,:,3] * prior_scaling[3])

        # cx/cy/w/h --> ymin/xmin/ymax/xmax
        bboxes = tf.stack([cy-h/2.0,cx-w/2.0,cy+h/2.0,cx+w/2.0], axis=-1)

        # shape为[batch_size, size, size, n_anchors, 4]
        return bboxes
    ####################################################################################################################

    ##################################【3】去除得分score<阈值threshold的解码得到边界框bboxes##############################
    # 从最后的特征图中筛选1次边界框bboxes原则(仅针对batchsize=1):最大类别得分score>阈值
    def _bboxes_select_layer(self,feature_predictions,feature_locations,anchor_bboxes,prior_scaling):
        # bboxes的个数=网络输出的shape之间的乘积
        n_bboxes = np.product(feature_predictions.get_shape().as_list()[1:-1])

        # 解码边界框位置location
        bboxes = self._bboxes_decode_layer(feature_locations,anchor_bboxes,prior_scaling)
        bboxes = tf.reshape(bboxes,[n_bboxes,4]) # [边界框bboxes数量，每个bbox的位置和大小]
        predictions = tf.reshape(feature_predictions,[n_bboxes,self.ssd_params.num_classes]) # [边界框bboxes数量，每个bbox的类别得分]
        # 移除背景的得分num_class预测值
        sub_predictions = predictions[:,1:]

        # 筛选最大的类别分数
        classes = tf.argmax(sub_predictions,axis=1) + 1 # 类别labels：最大的类别分数索引。(因为背景在第一个索引位置，故后面+1)
        scores = tf.reduce_max(sub_predictions,axis=1) # 最大类别得分max_class scores
        # ※※※筛选边界框bbox：最大类别得分>阈值(只用了第二个原则)※※※
        filter_mask = scores > self.threshold # 变成bool类型的向量：True留下、False去除
        classes = tf.boolean_mask(classes,filter_mask)
        scores = tf.boolean_mask(scores,filter_mask)
        bboxes = tf.boolean_mask(bboxes,filter_mask)

        return classes,scores,bboxes

    # 筛选所有的预测边界框：循环调用上面的筛选原则
    def _bboxes_select(self,predictions,locations):
        anchor_bboxes_list = self.anchors()
        classes_list = []
        scores_list = []
        bboxes_list = []

        # 对每个feature layer选择bboxes：循环调用上面的筛选原则
        for n in range(len(predictions)):
            anchor_bboxes = list(map(tf.convert_to_tensor,anchor_bboxes_list[n]))
            classes,scores,bboxes = self._bboxes_select_layer(predictions[n],locations[n],
                                                              anchor_bboxes,self.ssd_params.prior_scaling)
            classes_list.append(classes)
            scores_list.append(scores)
            bboxes_list.append(bboxes)
        # 整合所有的feature layer筛选的边界框结果
        classes = tf.concat(classes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        bboxes = tf.concat(bboxes_list, axis=0)

        return classes, scores, bboxes
    ####################################################################################################################

    def images(self):
        return self._images

    # 检测：得到预测边界框的类别、得分(概率)、边界框位置和大小
    def detections(self):
        return self._classes, self._scores, self._bboxes

if __name__ == '__main__':
    ssd = SSD()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, './SSD_model/ssd_vgg_300_weights.ckpt') # 导入模型