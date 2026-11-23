# 从采样点到频谱

给一段合成波形换一个采样率，哪些细节会消失？把不同频率的波叠加起来，又怎样读回它们的幅度和相位？下面四本笔记把这些问题拆成可以修改参数、重画图像的小实验。

| 笔记 | 可以验证什么 |
| --- | --- |
| [采样、量化与混叠](sampling-and-quantization.ipynb) | 给 2 Hz 的波形每秒采 20 个点，把同一个样本分别量化到 0.5 和 0.1 的间隔。 |
| [离散序列与时域滤波](discrete-systems.ipynb) | 对比补零与循环移位，观察两点移动平均如何削弱交替分量，再用延迟反馈合成衰减回声。 |
| [从向量投影到手算 DFT](projection-to-dft.ipynb) | 将四点信号投影到复指数基上，重建时域信号，并直接比较手算结果与 NumPy FFT。 |
| [频率轴、幅相谱与逆变换](frequency-and-phase.ipynb) | 找出 10 Hz 和 22 Hz 两个成分，核对单边谱上的幅度，使用复频谱还原波形，沿多通道张量的采样轴做变换。 |

所有输入都是合成序列。点开笔记可以直接看代码、图和输出。从本目录运行以下命令，可以自己调整采样率、量化间隔或反馈系数。

```sh
python -m pip install numpy matplotlib jupyterlab
python -m jupyterlab
```

[继续看四通道模拟 EEG 的切片与 Transformer 分类](../eeg-patches/README.md)
