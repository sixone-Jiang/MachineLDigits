# 这是一个基于sklearn原生手写数字数据集的测试



## 环境

预先准备： Windows环境、conda命令可用

为了避免环境冲突，我的建议是新建一个python环境

创建新的python环境：`conda create -n tensorflow python=3.7`

使用该环境：`conda activate tensorflow`

 其余按照`setup.txt`依次安装最新版本

## 运行

` python first.py `（注意需要用conda先激活tensor环境，且注意文件目录）

## 其他的一些解释

` data.py` 运行代码可以将原生数据集存储到本地（），（各数字真值保存在yanzheng.txt中）

## 文档中所用函数的解释

1. [归一化MinMaxScaler() ](https://blog.csdn.net/GentleCP/article/details/109333753)
2. [独热编码oneHot](https://www.cnblogs.com/zhoukui/p/9159909.html)
3. [分割数据集train_test_split](https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html)
4. [所使用的神经网络的类似解析](https://blog.csdn.net/yunfeather/article/details/106461462)
5. [model.evaluate](https://blog.csdn.net/qq_28979491/article/details/101529849)
6. [tensorflow官网](https://www.tensorflow.org/)

## 关于流程化

显然，这份代码训练的模型并没有保存和载入，后续将更新

## WHAT'S MORE?

这是一份不完善的文档，后续将继续修改

-- ALICE

# MachineLDigits
