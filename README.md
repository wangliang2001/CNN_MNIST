# MNIST Handwritten Digit Recognition

这个项目包含两个主要脚本：一个用于训练数字识别模型，另一个用于预测手写数字或MNIST数据集中的数字。

## main.py - 训练脚本

`main.py`是一个用于训练手写数字识别模型的Python脚本。它使用PyTorch框架，并在MNIST数据集上训练一个简单的卷积神经网络（CNN）。

### 特点

- 使用PyTorch框架构建和训练模型。
- 使用MNIST数据集进行训练。
- 支持模型保存，以便将来进行预测。

### 如何使用

运行脚本以训练模型：

```bash
python main.py
```

## predict.py - 预测脚本

`predict.py` 是一个用于预测手写数字的 Python 脚本。它可以使用 `main.py` 训练的模型来识别 MNIST 数据集中的数字或外部手写数字图片。

### 特点

- 使用训练好的模型进行数字预测。
- 支持从 MNIST 测试集或外部图片进行预测。
- 对外部图片进行预处理以符合 MNIST 风格。

### 如何使用

运行脚本并根据提示进行操作：

```bash
python predict.py
```
