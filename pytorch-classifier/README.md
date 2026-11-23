# 中文短句情感分类

用 8 条人工编写的短句训练词嵌入、平均池化和线性分类器，在 4 条重新组合的短句上检查预测。最后临时保存模型参数和词表，重新加载并比较两次预测。小样本用于练习训练和验证流程。

```sh
python -m pip install torch
python sentence_classifier.py
```
