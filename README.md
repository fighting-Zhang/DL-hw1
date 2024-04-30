# DL_hw1
神经网络和深度学习课程作业1：从零开始构建三层神经网络分类器，实现图像分类

1. 训练代码：`train.py`

    在代码中指定输出目录：`results_dir`和超参数字典`hyperparams`，运行后最优模型参数存储在`best_model_params.npz`，超参数字典存储在`hyperparams.json`。

    首次运行时，会下载Fashion-MNIST数据集到目录`./fashion-mnist`下。

2. 超参数网格搜索：`grid_search.py`

    可以尝试不同的超参数：初始学习率`initial_lrs`、学习率下降策略（下降方式、衰减系数等）`lr_strategy`、隐藏层大小`hidden_sizes`、L2正则化强度`l2_regs`，对训练结果的影响。


3. 测试代码：`test.py`

    在代码中指定测试目录：`results_dir`，加载该目录下保存的最优模型参数`best_model_params.npz`，和对应的模型结构超参数`hyperparams.json`。输出模型在测试集上的分类准确率(accuracy)。

4. 模型网络参数可视化：`visual.py`

    可视化隐藏层和输出层的模型权重：W1(hidden_sizes x 784)、W2(10 x hidden_sizes)。

    W1可视化为 `hidden_sizes` 张大小为28 x 28的图像；
    W2可视化为 10 张包含 `hidden_size` 个一维向量的图像。

5. 最优超参数组合和模型权重保存在：<a href="https://drive.google.com/drive/folders/1k04avH0DxMwTPwdTJ01kzwf-y_qR-5NM?usp=sharing">google drive</a>