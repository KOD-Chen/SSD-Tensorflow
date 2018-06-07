# SSD-Tensorflow<br>

## 声明：<br>
更详细的代码解读[Tensorflow实现SSD](https://zhuanlan.zhihu.com/p/37635878).<br>
欢迎关注[我的知乎](https://www.zhihu.com/people/chensicheng/posts).<br><br>

## 运行环境：<br>
Python3 + Tensorflow1.5 + OpenCV-python3.3.1 + Numpy1.13<br>
windows和ubuntu环境都可以<br><br>

## 准备工作：<br>
请在[SSD检测模型](https://pan.baidu.com/s/1snhuTsT)下载模型，并放到SSD_model文件夹下<br><br>

## 文件说明：<br>
### 1、ssd300_vgg.py：搭建ssd300网络模型，并解码网络输出<br>
·ssd_layers.py：SSD中各种网络层的定义<br>
·ssd_anchors.py：SSD的先验框anchor设置，与Caffe源码保持一致<br>
### 2、utils.py：功能函数：<br>
（1）预处理图片：白化、resize300x300、增加batchsize这一维度<br>
（2）处理/筛选边界框：<br>
    ·将边界框超出图片范围(0,0)-(300,300)的部分cut掉；<br>
    ·按类别置信度scores降序，对边界框进行排序并仅保留top_k=400；<br>
    ·计算IOU->采用NMS；<br>
    ·还原相对原图片的边界框真实位置和大小.<br>
### 3、drawbox.py：可视化最后的检测结果<br>
### 4、Main.py：SSD主函数：<br>
（1）搭建网络，解码网络输出并设置阈值去除得分低于阈值的边界框，得到边界框的类别、得分、位置大小<br>
（2）导入训练好的SSD模型<br>
（3）预处理图片-->处理/筛选边界框<br>
（4）可视化最后的检测结果<br>
### 5、SSD_data文件夹：<br>
包含待检测输入图片car.jpg、检测后的输出图片detection.jpg、ssd_300_vgg网络各个层的名称var_name.txt<br><br>

## 运行Main.py即可得到效果图：<br>
1、car.jpg：输入的待检测图片<br><br>
![image](https://github.com/KOD-Chen/SSD-Tensorflow/blob/master/SSD_data/car.jpg)<br><br>
2、detection.jpg：检测结果可视化<br><br>
![image](https://github.com/KOD-Chen/SSD-Tensorflow/blob/master/SSD_data/detection.jpg)<br>
