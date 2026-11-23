"""从 RMSNorm 和 SwiGLU 构造 pre-norm 解码器层。"""

import torch
import torch.nn.functional as F

from attention import multi_head_attention_rope


def rms_norm(h, weight, eps=1e-5):
    """
    h:      [B, T, d]
    weight: [d]

    返回：
    out: [B, T, d]
    """

    # 1. 对每个特征平方。
    squared = h.square()               # [B, T, d]

    # 2. 每个位置分别计算特征的平方均值。
    mean_square = squared.mean(
        dim=-1,
        keepdim=True,
    )                                  # [B, T, 1]

    # 3. 加上小常数，再开平方。
    rms = torch.sqrt(mean_square + eps) # [B, T, 1]

    # 4. 每个位置除以自己的 RMS。
    normalized = h / rms               # [B, T, d]

    # 5. 逐特征乘可训练的缩放参数。
    out = normalized * weight          # [B, T, d]

    return out


def swiglu_ffn(h, W_gate, W_up, W_down):
    """
    不含偏置的 SwiGLU 前馈网络。

    h:      [B, T, d]
    W_gate: [d, d_ff]
    W_up:   [d, d_ff]
    W_down: [d_ff, d]

    返回：
    out: [B, T, d]
    """

    # 1. 门控分支的线性投影。
    gate_pre = h @ W_gate          # [B, T, d_ff]

    # 2. 另一条分支的线性投影。
    up = h @ W_up                  # [B, T, d_ff]

    # 3. 非线性门控系数。
    gate = F.silu(gate_pre)        # [B, T, d_ff]

    # 4. 逐元素调制特征。
    hidden = gate * up            # [B, T, d_ff]

    # 5. 映射回模型宽度。
    out = hidden @ W_down         # [B, T, d]

    return out


def transformer_block(h, params, num_heads):
    """
    一个 pre-norm Transformer block。

    h: [B, T, d]
    params: 当前这个 block 的参数字典

    返回：
    h:       [B, T, d]
    weights: [B, num_heads, T, T]
    """

    # ---------- 注意力分支 ----------

    # 1. 归一化分支输入，同时保留原来的 h。
    u = rms_norm(
        h,
        params["norm1_weight"],
    )

    # 2. 计算注意力更新。
    attn_out, weights = multi_head_attention_rope(
        u,
        params["W_q"],
        params["W_k"],
        params["W_v"],
        params["W_o"],
        num_heads,
    )

    # 3. 第一次残差更新。
    h = h + attn_out

    # ---------- FFN 分支 ----------

    # 4. 对更新后的 h 再做一次归一化。
    u = rms_norm(
        h,
        params["norm2_weight"],
    )

    # 5. 计算 FFN 更新。
    ffn_out = swiglu_ffn(
        u,
        params["W_gate"],
        params["W_up"],
        params["W_down"],
    )

    # 6. 第二次残差更新。
    h = h + ffn_out

    return h, weights
