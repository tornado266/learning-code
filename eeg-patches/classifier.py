import torch
from torch import nn
from torch.nn import functional as F

from eeg_patches import EEGPatchEmbedding


class EEGClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_patches,
        patch_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        num_classes,
    ):
        super().__init__()

        # 1. EEG → token 向量
        # 嵌入层的构成：
        # 切片、线性投影、通道 embedding、时间 embedding
        self.patch_embedding = EEGPatchEmbedding(
            num_channels=num_channels,
            num_patches=num_patches,
            patch_size=patch_size,
            d_model=d_model,
        )

        # 2. 创建多层 Encoder
        # 每次循环都新建一层，因此各层拥有独立的参数
        self.blocks = nn.ModuleList()

        for _ in range(num_layers):
            block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.blocks.append(block)

        # 3. 在所有 Encoder 层之后，再进行一次归一化
        # 对每个 token 的最后 d_model 个特征进行处理
        self.final_norm = nn.LayerNorm(d_model)

        # 4. 整段 EEG 的特征 → 类别分数
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x：[B, C, S]

        # EEG → token 向量
        h = self.patch_embedding(x)
        # h：[B, N, d_model]

        # 窗口内双向注意力：
        # 这里没有传入因果 mask
        for block in self.blocks:
            h = block(h)

        # 形状仍为 [B, N, d_model]
        h = self.final_norm(h)

        # 沿 token 维求平均，得到每条样本的整体表示
        features = h.mean(dim=1)
        # features：[B, d_model]

        # 输出各类别的原始分数
        logits = self.classifier(features)
        # logits：[B, num_classes]

        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    model = EEGClassifier(
        num_channels=4,
        num_patches=8,
        patch_size=32,
        d_model=16,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=3,
    )

    # 两条样本，每条 4 个通道，每通道 256 个采样点
    # 这里只用随机数据验证结构
    x = torch.randn(2, 4, 256)

    model.eval()

    with torch.no_grad():
        logits = model(x)

        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

    print("输入形状：", x.shape)
    print("类别分数形状：", logits.shape)
    print("类别概率形状：", probabilities.shape)
    print("预测类别形状：", predictions.shape)

    print("每条样本的概率之和：", probabilities.sum(dim=-1))

    model.train()

    # 演示标签格式：
    # 第 0 条样本属于类别 0，第 1 条样本属于类别 2
    # 真实实验中必须换成数据对应的真实标签
    labels = torch.tensor([0, 2], dtype=torch.long)

    # 重新前向计算，保留计算图
    logits = model(x)

    # 直接传入原始 logits
    loss = F.cross_entropy(logits, labels)

    print("loss：", loss.item())

    assert logits.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(loss)
