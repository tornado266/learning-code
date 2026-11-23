# 多通道 EEG 切片与嵌入

PyTorch 学习实验。把 2 条模拟样本中每个通道的 256 个采样点切成 8 段，经线性投影后加上通道和时间嵌入。4 个通道共得到 32 个 16 维向量。示例用随机数验证输入输出形状。

```sh
python -m pip install torch
python eeg_patches.py
```
