# 人体骨骼关键点

人体骨骼关节点对于描述人体姿态、预测人体行为至关重要，因此人体骨骼关节点检测是诸多计算机视觉任务的基础，例如动作分类、异常行为检测、以及自动驾驶等。

本代码库是如下论文的实现：

    @InProceedings{cao2017realtime,
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
      }

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)


# 数据集

使用 AI Challenger 2017 的人体骨骼关键点数据集，包含30万张图片，70万人。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。

每个人物的全部人体骨骼关键点共有14个，编号顺序如表所示，依次为：

|1/右肩|2/右肘|3/右腕|4/左肩|5/左肘|
|---|---|---|---|---|
|6/左腕|7/右髋|8/右膝|9/右踝|10/左髋|
|11/左膝|12/左踝|	13/头顶|14/脖子|

 ![image](https://github.com/foamliu/Keypoints/raw/master/images/keypoint-example.png)

下载点这里：[人体骨骼关键点数据集](https://challenger.ai/datasets/keypoint)，放在 data 目录下。

## 部位置信图(Part Confidence Maps)

原图 | 部位置信图 | 部位 |
|---|---|---|
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_0.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_0.png) | RShoulder |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_1.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_1.png) | RElbow |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_2.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_2.png) | RWrist |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_3.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_3.png) | LShoulder |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_4.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_4.png) | LElbow |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_5.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_5.png) | LWrist |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_6.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_6.png) | RHip |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_7.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_7.png) | RKnee |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_8.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_8.png) | RAnkle |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_9.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_9.png) | LHip |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_10.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_10.png) | LKnee |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_11.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_11.png) | LAnkle |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_12.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_12.png) | Head |
|![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_image_13.png)  | ![image](https://github.com/foamliu/Keypoints/raw/master/images/datav_heatmap_13.png) | Neck |

## 部位亲和场(Part Affinity Fields)


# 网络结构

 ![image](https://github.com/foamliu/Keypoints/raw/master/images/two-branch_multi-stage_CNN.png)

# 用法

## 数据预处理
提取210,000 张训练图片和30,000 张验证图片：
```bash
$ python pre-process.py
```

## 训练
```bash
$ python train.py
```

可视化训练过程，执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

## 演示
下载 [预训练模型](https://github.com/foamliu/Keypoints/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录，然后执行:

```bash
$ python demo.py
```

# 鸣谢
本代码主要基于 @michalfaber 的代码库 keras_Realtime_Multi-Person_Pose_Estimation。
