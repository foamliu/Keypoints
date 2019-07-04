# Human KeyPoints

Human KeyPoints in PyTorch

## Dependencies
- PyTorch 1.0

# Dataset

使用 AI Challenger 2017 的人体骨骼关键点数据集，包含30万张图片，70万人。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。

每个人物的全部人体骨骼关键点共有14个，编号顺序如表所示，依次为：

|1/右肩|2/右肘|3/右腕|4/左肩|5/左肘|
|---|---|---|---|---|
|6/左腕|7/右髋|8/右膝|9/右踝|10/左髋|
|11/左膝|12/左踝|	13/头顶|14/脖子|

 ![image](https://github.com/foamliu/Keypoints/raw/master/images/keypoint-example.png)

下载点这里：[人体骨骼关键点数据集](https://challenger.ai/datasets/)，放在 data 目录下。

# Architecture

 ![image](https://github.com/foamliu/Keypoints/raw/master/images/two-branch_multi-stage_CNN.png)

# Usage

## 数据预处理
提取210,000 张训练图片和30,000 张验证图片：
```bash
$ python pre-process.py
```


## Train
```bash
$ python train.py
```

可视化训练过程，执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

## Demo
Download pre-trained then run:

```bash
$ python demo.py
```


