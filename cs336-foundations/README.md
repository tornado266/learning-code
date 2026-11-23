# CS336 · 分词与训练成本

给定一段文本，能不能学出一套合并规则，用它编码，再把空格、换行和中文逐个还原？有了 token，跑一次线性层又要用多少内存、做多少次运算？下面三个入口分别保留了训练过程、完整分词器和张量实验。

| 直接打开 | 可以观察什么 |
| --- | --- |
| [BPE 分步笔记](bpe_workbook.ipynb) | 从相邻字节计数开始，练习合并、预分词、频次相同时的选择，以及特殊标记和多段输入的处理。 |
| [可运行的分词器](bpe_review.py) | 用小语料训练词表，编码混合文本；将状态写入临时文件，重载后检查 ID 与解码结果。 |
| [训练系统实验](training_systems.ipynb) | 从嵌入查表开始，核算线性层、算术强度和内存带宽，比较浮点格式、张量布局、梯度累积与激活重计算。 |

### 拆开一次线性层

输入有 3 条序列，每条 5 个位置，每个位置用 8 个数表示。乘上一个 `8 × 16` 的权重矩阵后，得到 `3 × 5 × 16` 的输出。用 FP32 保存时，128 个权重占 512 字节，240 个输出占 960 字节。按一次乘加计两次浮点运算，这次前向约需 **3,840 FLOPs**。

Notebook 后半段把 8 条样本一次更新，与分成 4 个等大的微批次累积梯度再更新进行对照。最后一个实验把同一组参数送入普通前向和激活重计算，逐项核对两层权重的梯度。

### 复现

从仓库根目录执行以下命令。分词器仅依赖 `regex`，训练系统实验还需要 PyTorch 和 einops。

```sh
python -m pip install regex
python cs336-foundations/bpe_review.py
```

```sh
python -m pip install torch "einops>=0.6.1" jupyterlab ipykernel
jupyter lab cs336-foundations/training_systems.ipynb
```

[继续看因果注意力、RoPE 和完整 Transformer](../cs336-transformer/README.md)
