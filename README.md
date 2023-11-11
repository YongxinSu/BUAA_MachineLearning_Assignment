# BUAA_MachineLearning_Assignment

## Usage
1. Download datasets from https://github.com/yanglixiaoshen/ML-Class-Assignment.git .
```
data
- 3-Saliency-TestSet
- 3-Saliency-TrainSet
```
2. Creating conda environment.
```bash
conda env create -f requirements.yml
conda activate torch
```
3. Training model 
- You can edit `config.yml` to change hyperparameters before training.
```bash
python train.py --config ./config.yml --data ./data 
``` 
- Using the following script you can view the training process real time.
```bash
tensorboard --logdir ./
```

4. Model inference and testing
You must specify the checkpoint files(parameter `--model`) for model inference.
```bash
python inference.py --config ./config.yml --data ./data --model ./checkpoint/checkpoint_epoch1.pth
```

## feature development

## 数据增强

## 数据扩充

## 损失函数

## VIT

## 通道空间注意力



