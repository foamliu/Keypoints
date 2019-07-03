# Human KeyPoints

Human KeyPoints in PyTorch

## Dependencies
- PyTorch 1.0

# Dataset

 ![image](https://github.com/foamliu/Keypoints/raw/master/images/keypoint-example.png)

Download：[Dataset](https://challenger.ai/) into data folder.

# Architecture

 ![image](https://github.com/foamliu/Keypoints/raw/master/images/two-branch_multi-stage_CNN.png)

# Usage

## Data Pre-processing
Extract training images:
```bash
$ python pre-process.py
```

## Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

## Demo
Download pre-trained model then run:

```bash
$ python demo.py
```


