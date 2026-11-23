# 多通道 EEG 切片与 Transformer 分类

输入是一批四通道模拟信号，每条样本有两秒，每个通道以 128 Hz 采样。三个脚本使用同一套输入规格，可以直接比较不同的切片方式，以及模型如何把一段信号映射为三个类别分数。

| 实现 | 单条样本的输入与输出 |
| --- | --- |
| [每个通道独立切片](eeg_patches.py) | 每个通道的 256 个点分成 8 片。每片的 32 个点经共享线性层投影，再加通道与时间嵌入，得到 32 个 16 维向量。 |
| [同一时段合并四个通道](joint_patches.py) | 一个时间片收集 4 × 32 个点，128 个数共同投影。相同输入得到 8 个 16 维向量。 |
| [窗口级 Transformer 分类器](classifier.py) | 两层双向 Encoder 在窗口内交换信息，对 32 个向量做归一化和平均池化，最后给出 3 个类别分数。 |

分类器的运行示例会输出 logits 的形状、每行概率之和，以及两条演示标签对应的交叉熵。随机输入用于检验计算过程。

```sh
python -m pip install torch
python eeg_patches.py
python joint_patches.py
python classifier.py
```

[查看采样、滤波、频谱与 DFT 实验](../eeg-signal-lab/README.md)
