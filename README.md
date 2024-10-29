## BiG2S + Discrete CVAE
## Reproduce the Results
### 1. Environmental setup
Please ensure that conda has been properly initialized, i.e. **conda activate** is runnable. Then
准备 conda 环境和相关的工具包
```
conda create -n retrog2s python==3.9.19
conda activate retrog2s
pip install -r requirements.txt
```

### 2. 反应数据预处理
从原始数据集如uspto-50k中预先提取产物图结构和反应物序列（字符串）结构。$DATASET 用具体的反应数据集名字替换，如uspto-50k。
```
python preprocess.py --dataset_name $DATASET
```
这里 $DATASET 可以用[**uspto_50k**, **uspto_diverse**, **uspto_full**] 中的指定一个来替换。

### 3. Model training and validation
Run the training script by
```
export CUDA_VISIBLE_DEVICES=1 # 数字代表显卡的id,一般默认为0
python train.py --dataset_name $DATASET
```
这里 $DATASET 可以用[**uspto_50k**, **uspto_diverse**, **uspto_full**] 中的指定一个来替换，同上。

Optionally, run the evaluation script by
```
python predict.py --dataset_name $DATASET
```
这里 $DATASET 的选取形式同上。

## Acknowledgement
We refer to the code of [RetroDCVAE](https://github.com/MIRALab-USTC/DD-RetroDCVAE) and [BiG2S](https://github.com/AILBC/BiG2S). Thanks for their contributions.

