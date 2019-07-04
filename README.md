# Person Keypoint Detection

## Performance
- Test with method whole image.
- SAD normalized by 1000.
- Input img is normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
- Both erode and dialte of alpha for trimap

|Methods|SAD|MSE|Update|
|---|---|---|---|
|Encoder-Decoder network|40.7|0.014|

## Dependencies
- PyTorch 1.0

# Dataset

![image](https://github.com/foamliu/Keypoints/raw/master/images/keypoint-example.png)

Download: [Dataset](https://challenger.ai/) into data folder.

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


