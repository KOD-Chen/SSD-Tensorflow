# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/6/4$ 17:15$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : Main$.py
# Description :SSD主函数.
# --------------------------------------

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from SSD.ssd300_vgg import SSD
from SSD.utils import preprocess_image,process_bboxes
from SSD.drawbox import plt_bboxes

def main():
    # 【1】搭建网络-->解码网络输出-->设置图片的占位节点
    ssd_net = SSD() # 搭建网络：ssd300_vgg
    classes, scores, bboxes = ssd_net.detections() # 设置分数阈值，解码网络输出得到bbox的类别、得分(概率)、边界框位置和大小
    images = ssd_net.images() # 设置图片的占位节点：images是一个tf.placeholder

    # 【2】导入SSD模型
    sess = tf.Session()
    ckpt_filename = './SSD_model/ssd_vgg_300_weights.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    # 【3】预处理图片-->处理预测边界框bboxes
    img = cv2.imread('./SSD_data/car.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 预处理图片：
    # 1、白化；
    # 2、resize300*300；
    # 3、增加batchsize这个维度.
    img_prepocessed = preprocess_image(img)
    # 将预处理好的图片赋给图片的占位节点
    rclasses, rscores, rbboxes = sess.run([classes, scores, bboxes], feed_dict={images: img_prepocessed})
    # 处理预测边界框：
    # 1、cut the box:将边界框超出整张图片(0,0)—(300,300)的部分cut掉；
    # 2、按类别置信度scores降序，对边界框进行排序并仅保留top_k=400；
    # 3、计算IOU-->NMS;
    # 4、根据先验框anchor调整预测边界框的大小.
    rclasses, rscores, rbboxes = process_bboxes(rclasses, rscores, rbboxes)

    # 【4】可视化最终的检测结果
    plt_bboxes(img, rclasses, rscores, rbboxes)
    print('SSD detection has done!')

if __name__ == '__main__':
    main()