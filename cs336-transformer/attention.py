"""单头、多头及带旋转位置编码的因果自注意力。"""

import torch


def single_head_attention(h, W_q, W_k, W_v, W_o):
    """
    单头因果自注意力。

    h:   [B, T, d]
    W_q: [d, d_k]
    W_k: [d, d_k]
    W_v: [d, d_v]
    W_o: [d_v, d]

    返回：
    out:     [B, T, d]，注意力模块的输出
    weights: [B, T, T]，用于观察的注意力权重
    """

    # 1. 同一份输入，分别生成查询、键和值。
    q = h @ W_q                       # [B, T, d_k]
    k = h @ W_k                       # [B, T, d_k]
    v = h @ W_v                       # [B, T, d_v]

    # 2. 计算每个查询位置对每个来源位置的分数。
    # transpose 只交换最后两个维度。
    scores = q @ k.transpose(-2, -1)  # [B, T, T]

    # 3. 按查询/键的维度缩放分数。
    d_k = q.shape[-1]
    scores = scores / (d_k ** 0.5)

    # 4. 构造因果掩码：True 标记未来位置。
    T = h.shape[1]
    future_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=h.device),
        diagonal=1,
    )

    # 将未来位置的分数替换为负无穷。
    masked_scores = scores.masked_fill(
        future_mask,
        float("-inf"),
    )

    # 5. 在来源位置之间分配权重。
    weights = torch.softmax(masked_scores, dim=-1)

    # 6. 按权重汇总各位置的 Value 向量。
    context = weights @ v             # [B, T, d_v]

    # 7. 将汇总后的特征映射回模型宽度。
    out = context @ W_o               # [B, T, d]

    return out, weights


def multi_head_attention(
    h, W_q, W_k, W_v, W_o, num_heads
):
    """
    多头因果自注意力。

    h: [B, T, d]

    W_q、W_k、W_v、W_o: [d, d]

    返回：
    out:     [B, T, d]
    weights: [B, num_heads, T, T]
    """

    B, T, d = h.shape

    # 检查总宽度能否均分给各个头。
    assert d % num_heads == 0
    d_head = d // num_heads

    # 1. 一次投影，得到所有头的 Q、K、V。
    q = h @ W_q                         # [B, T, d]
    k = h @ W_k                         # [B, T, d]
    v = h @ W_v                         # [B, T, d]

    # 2. 拆出头维，再把头维移到序列维前面。
    q = q.reshape(
        B, T, num_heads, d_head
    ).transpose(1, 2)

    k = k.reshape(
        B, T, num_heads, d_head
    ).transpose(1, 2)

    v = v.reshape(
        B, T, num_heads, d_head
    ).transpose(1, 2)

    # 此时 Q、K、V：
    # [B, num_heads, T, d_head]

    # 3. 每个头分别计算缩放点积分数。
    scores = q @ k.transpose(-2, -1)
    scores = scores / (d_head ** 0.5)
    # scores：[B, num_heads, T, T]

    # 4. 屏蔽未来位置。
    future_mask = torch.triu(
        torch.ones(
            T, T,
            dtype=torch.bool,
            device=h.device,
        ),
        diagonal=1,
    )

    scores = scores.masked_fill(
        future_mask,
        float("-inf"),
    )

    # 5. 每个头分别在来源位置之间分配权重。
    weights = torch.softmax(scores, dim=-1)

    # 6. 每个头分别汇总 Value。
    head_outputs = weights @ v
    # [B, num_heads, T, d_head]

    # 7. 恢复位置优先的排列，再拼接各个头。
    context = head_outputs.transpose(1, 2)
    context = context.reshape(B, T, d)

    # 8. 输出投影，组合各个头提取的特征。
    out = context @ W_o

    return out, weights


def apply_rope(x, base=10000.0):
    """
    对分头后的 Q 或 K 应用 RoPE。

    x: [B, num_heads, T, d_head]

    使用相邻特征配对：
    (0,1)、(2,3)、……

    返回形状与输入相同。
    """

    B, num_heads, T, d_head = x.shape

    assert d_head % 2 == 0

    # 1. 序列中的位置编号：[0, 1, ..., T-1]
    positions = torch.arange(
        T,
        device=x.device,
        dtype=torch.float32,
    )

    # 2. 每一对特征的起始下标：[0, 2, 4, ...]
    pair_dims = torch.arange(
        0, d_head, 2,
        device=x.device,
        dtype=torch.float32,
    )

    # 每对特征的旋转频率。
    inv_freq = base ** (-pair_dims / d_head)

    # 3. 位置 × 频率，得到角度表。
    # [T, 1] * [1, d_head/2]
    # → [T, d_head/2]
    angles = (
        positions.reshape(T, 1)
        * inv_freq.reshape(1, d_head // 2)
    )

    # 4. 增加 batch、head 广播维。
    # [1, 1, T, d_head/2]
    cos = angles.cos().reshape(
        1, 1, T, d_head // 2
    ).to(dtype=x.dtype)

    sin = angles.sin().reshape(
        1, 1, T, d_head // 2
    ).to(dtype=x.dtype)

    # 5. 取出每对特征中的第一项和第二项。
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # 6. 对每一对特征执行二维旋转。
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # 7. 恢复相邻交错的特征排列。
    paired = torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    )
    # [B, num_heads, T, d_head/2, 2]

    return paired.reshape(B, num_heads, T, d_head)


def multi_head_attention_rope(
    h, W_q, W_k, W_v, W_o,
    num_heads,
    base=10000.0,
):
    """
    带 RoPE 的多头因果自注意力。

    h: [B, T, d]

    返回：
    out:     [B, T, d]
    weights: [B, num_heads, T, T]
    """

    B, T, d = h.shape

    assert d % num_heads == 0
    d_head = d // num_heads

    # 相邻二维配对要求每个头的宽度为偶数。
    assert d_head % 2 == 0

    # 1. 投影。
    q = h @ W_q
    k = h @ W_k
    v = h @ W_v

    # 2. 分头。
    q = q.reshape(
        B, T, num_heads, d_head
    ).transpose(1, 2)

    k = k.reshape(
        B, T, num_heads, d_head
    ).transpose(1, 2)

    v = v.reshape(
        B, T, num_heads, d_head
    ).transpose(1, 2)

    # 3. 新增：给 Q、K 加入旋转位置信息。
    q = apply_rope(q, base=base)
    k = apply_rope(k, base=base)

    # 4. 用旋转后的 Q、K 计算匹配分数。
    scores = q @ k.transpose(-2, -1)
    scores = scores / (d_head ** 0.5)

    # 5. 因果掩码。
    future_mask = torch.triu(
        torch.ones(
            T, T,
            dtype=torch.bool,
            device=h.device,
        ),
        diagonal=1,
    )

    scores = scores.masked_fill(
        future_mask,
        float("-inf"),
    )

    # 6. 权重和信息汇总。
    weights = torch.softmax(scores, dim=-1)
    head_outputs = weights @ v

    # 7. 合头与输出投影。
    context = head_outputs.transpose(1, 2)
    context = context.reshape(B, T, d)

    out = context @ W_o

    return out, weights
