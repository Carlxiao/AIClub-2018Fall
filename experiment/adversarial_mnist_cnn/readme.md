# 在MNIST数据集生成CNN的对抗样本

## 1. 主要思路

先实现一个简单的CNN，在MNIST数据集上训练，达到较高的99%以上的验证集准确率。然后，选定一张训练集中的图片以及目标对抗样本的错误分类，学习一个加在图片上的噪声，使得加了噪声的图片被网络预测为选定的错误分类（网络最后的softmax输出与错误的onehot标签尽可能接近）。在对抗样本的训练过程中，需要固定住网络的参数，唯一的可学习变量是噪声。

## 2. 实验过程

实验代码使用TensorFlow 1.11以及[Tensorpack](https://github.com/tensorpack/tensorpack)编写。

### 2.1 在MNIST数据集训练一个简单的CNN

CNN的主要框架参考了[Keras中的样例代码](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)，共有两个卷积层、一个池化层、两个全连接层以及最后的softmax，其中两个全连接层之前都加入dropout减轻过拟合。误差函数为softmax输出与onehot标签的交叉熵。训练时，选择批大小为128，使用Adam优化器和1e-3的初始学习率。最终，验证集准确率能够达到99.33%。

代码：[train.py](train.py)

### 2.2 生成对抗样本

从训练数据中选出16张图片，训练分别加在这些图片上的噪声（一个随机初始化的可学习变量），使得加噪图片被预测为错误的标签（选择9-d作为数字d的错误标签）。在这一过程中，网络框架不变，但需要固定住所有网络参数，并去掉dropout。损失函数与训练CNN时的一样，只是监督标签变为了选定的错误标签。为了使生成的对抗样本尽量接近原图，还需要为噪声加上正则误差。正则函数的选取对生成的对抗样本有显著的影响，见结果展示部分。

代码：[adversarial.py](adversarial.py)

## 3. 结果展示

### 3.1 对噪声使用L2正则

训练结果：所有对抗样本都被识别为设定的错误标签。

> pred_label: [7 1 3 0 5 9 0 8 8 7 5 6 7 2 6 1]
L1Norm: 20.069
L2Norm: 0.39969
cross_entropy_loss: 0.010374
regularize_cost: 0.20448
train_error: 0

生成的对抗样本及对应的噪声：

![mnist_adv_sample_l2](/assets/mnist_adv_sample_l2.png) ![mnist_adv_sample_l2_noise](/assets/mnist_adv_sample_l2_noise.png)


### 3.2 对噪声使用L1正则

训练结果：除了第12个样本（数字3）仍然被识别为3外，其他都被识别为选定的错误标签。

> pred_label: [7 1 3 0 5 9 0 8 8 7 5 3 7 2 6 1]
L1Norm: 8.4974
L2Norm: 0.54432
cross_entropy_loss: 0.18368
regularize_cost: 1.3596
train_error: 0.0625


生成的对抗样本及对应的噪声：

![mnist_adv_sample_l1](/assets/mnist_adv_sample_l1.png) ![mnist_adv_sample_l1_noise](/assets/mnist_adv_sample_l1_noise_3vkgr0xok.png)

### 3.3 对噪声不使用正则

训练结果：所有样本都被识别为设定的错误标签，并且损失函数（交叉熵）几乎为0。

> pred_label: [7 1 3 0 5 9 0 8 8 7 5 6 7 2 6 1]
L1Norm: 85.013
L2Norm: 1.2005
cross_entropy_loss: 2.1642e-05
regularize_cost: 1.8447
train_error: 0

生成的对抗样本和对应的噪声：

![mnist_adv_sample_unreg](/assets/mnist_adv_sample_unreg.png)  ![mnist_adv_sample_unreg_noise](/assets/mnist_adv_sample_unreg_noise.png)

### 3.4 对比

噪声正则函数的使用和选取对对抗样本的生成有着显著的影响。

- 使用L2正则时，训练过程中噪声的一范数和二范数都会下降，生成的噪声相对不明显（幅值较小），但分布较广，因此对抗样本看起来有一定的模糊。
- 使用L1正则时，训练过程中噪声的一范数下降，而二范数则会上升。生成的噪声相对稀疏，但存在较多高亮度的噪点。
- 不使用正则时，训练过程中噪声的一范数和二范数都会上升。网络为了以尽可能高的置信度将样本预测为设定的错误标签，让噪声变得很大，生成的一些对抗样本可能人眼也不好区分（如第11个数字4）。
