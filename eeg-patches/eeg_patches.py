import torch
from torch import nn

class EEGPatchEmbedding(nn.Module):
    def __init__(
        self,
        num_channels,
        num_patches,
        patch_size,
        d_model,
    ):
        super().__init__()

        # 保存当前实例采用的输入规格
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.d_model = d_model

        # 将每个片段的K个采样值，映射成d_model维向量
        # 所有通道、所有时间片共享这个线性层
        self.projection = nn.Linear(patch_size, d_model)

        # 每个通道、时间一个可学习向量
        self.channel_embedding = nn.Embedding(
        num_channels, d_model
        )

        self.time_embedding = nn.Embedding(num_patches, d_model)

    def forward(self, x):
        # x: [B,C,S]
        B,C,S = x.shape
        K = self.patch_size

        # 这个教学版本采用固定通道数量和固定通道长度
        assert C == self.num_channels
        assert S % K == 0

        P = S // K
        assert P == self.num_patches

        # 1.每个通道分别切成P个时间片
        # [B, C, S] -> [B, C, P, K]
        patches = x.reshape(B, C, P, K)


        # 2. 编码每个片段的波形
        # [B, C, P, K] → [B, C, P, d_model]
        tokens = self.projection(patches)

        # 3.准备通道ID和时间片ID
        channel_ids = torch.arange(C, device=x.device)
        time_ids = torch.arange(P, device=x.device)

        # [C, d_model] —— [1, C, 1, d_model)
        channel_info = self.channel_embedding(
        channel_ids
        ).reshape(1, C, 1, self.d_model)

        # [P, d_model] —— [1, 1, P, d_model)
        time_info = self.time_embedding(
        time_ids
        ).reshape(1, 1, P, self.d_model)

        # 4.通过广播，添加通道和时间信息
        # 三者相加后仍为{B, C, P, d_model)
        tokens = tokens + channel_info + time_info

        # 5. 把“通道 *时间片”展开成token序列
        # [B, C, P, d_model] → [B, C * P, d_model]
        tokens = tokens.reshape(B, C * P, self.d_model)

        return tokens


if __name__ == "__main__":
    torch.manual_seed(0)

    # 输入规格
    sampling_rate = 128
    duration_seconds = 2

    B = 2
    C = 4
    S = sampling_rate * duration_seconds

    patch_size = 32
    num_patches = S // patch_size
    d_model = 16

    # 随机数只用于检查代码和形状
    x = torch.randn(B, C, S, dtype=torch.float32)

    patch_embedding = EEGPatchEmbedding(
        num_channels=C,
        num_patches=num_patches,
        patch_size=patch_size,
        d_model=d_model,
    )

    h = patch_embedding(x)

    print("原始 EEG：", x.shape)
    print("token 表示：", h.shape)
    print("投影权重：", patch_embedding.projection.weight.shape)
    assert h.shape == (B, C * num_patches, d_model)
