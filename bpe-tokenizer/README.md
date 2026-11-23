# 字节级 BPE 分词器

CS336 自学练习。统计相邻字节对，逐轮合并，编码时保留预分词边界和特殊标记。示例用一小段文本训练，再用含中文、换行和特殊标记的文本验证可还原性。

```sh
python -m pip install regex
python tokenizer.py
```
