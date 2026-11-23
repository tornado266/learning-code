"""统计当前独立输入/输出权重配置的参数量与前向矩阵乘法开销。"""


def architecture_report(
    vocab_size,
    d,
    d_ff,
    num_heads,
    num_layers,
    B,
    T,
):
    # 当前多头注意力和 RoPE 的尺寸要求。
    assert num_heads > 0
    assert d % num_heads == 0

    d_head = d // num_heads
    assert d_head % 2 == 0

    # ---------- 参数量 ----------

    block_params = (
        4 * d * d
        + 3 * d * d_ff
        + 2 * d
    )

    total_params = (
        2 * vocab_size * d
        + num_layers * block_params
        + d
    )

    # ---------- 每层主要矩阵乘法 FLOPs ----------

    attention_projection_flops = 8 * B * T * d * d

    attention_mix_flops = 4 * B * T * T * d

    ffn_flops = 6 * B * T * d * d_ff

    block_flops = (
        attention_projection_flops
        + attention_mix_flops
        + ffn_flops
    )

    # ---------- 完整模型前向 FLOPs ----------

    output_flops = 2 * B * T * d * vocab_size

    total_flops = (
        num_layers * block_flops
        + output_flops
    )

    # 单层一张注意力权重张量的元素数。
    attention_weight_elements = (
        B * num_heads * T * T
    )

    return {
        "每头宽度": d_head,
        "总参数量": total_params,
        "前向主要矩阵乘法 FLOPs": total_flops,
        "每层注意力权重表元素数": attention_weight_elements,
    }
