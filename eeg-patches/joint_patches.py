"""对照实验：把同一时间片的所有通道一起投影。"""

import torch
from torch import nn


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, S = 2, 4, 256
    patch_size = 32
    d_model = 16
    x = torch.randn(B, C, S)

    P = S // patch_size

    patches = x.reshape(B, C, P, patch_size)

    # 将同一时间片的所有通道放在一起
    joint_patches = patches.permute(
        0, 2, 1, 3
    ).reshape(B, P, C * patch_size)

    joint_projection = nn.Linear(
        C * patch_size, d_model
    )

    joint_h = joint_projection(joint_patches)

    print(joint_patches.shape)
    print(joint_h.shape)

    assert joint_patches.shape == (2, 8, 128)
    assert joint_h.shape == (2, 8, 16)
