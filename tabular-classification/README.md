# 表格数据二分类

该学习实验使用 scikit-learn 自带的 Wisconsin Diagnostic Breast Cancer 数据集（569 条记录、30 个数值特征），通过 TensorFlow 比较逻辑回归和两层全连接网络。按类别分层划分训练、验证和测试集（398 / 85 / 86），用训练集拟合标准化器。模型以恶性为正类。Notebook 包括损失与准确率曲线、混淆矩阵、阈值和一例误判的线性特征贡献。

| 模型 | 准确率 | 正类召回率 | F1 | 假阴性 |
| --- | ---: | ---: | ---: | ---: |
| 逻辑回归 | 0.9884 | 0.9688 | 0.9841 | 1 |
| 两层全连接网络 | 0.9767 | 0.9375 | 0.9677 | 2 |

以上数值来自原笔记保存的一次固定划分。探索模型期间已查看这组测试结果，比较仅反映本练习的条件。

在此目录执行 `python -m pip install -r requirements.txt`，再运行 `jupyter lab classification.ipynb`，从头执行全部单元。公开数据由 scikit-learn 随包提供，无需下载用户数据。

[scikit-learn 数据说明](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) · [UCI Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
