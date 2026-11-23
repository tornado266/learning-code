"""两层示例使用的解码器语言模型与单序列贪心生成。"""

import torch
from torch import nn

from layers import rms_norm, transformer_block


class TransformerBlock(nn.Module):
    def __init__(self, d, d_ff, num_heads):
        # 初始化 PyTorch 的模块管理机制。
        super().__init__()

        self.num_heads = num_heads

        # 当前这个 block 自己拥有的参数。
        self.params = nn.ParameterDict({
            "norm1_weight": nn.Parameter(
                torch.ones(d)
            ),
            "norm2_weight": nn.Parameter(
                torch.ones(d)
            ),

            "W_q": nn.Parameter(
                0.1 * torch.randn(d, d)
            ),
            "W_k": nn.Parameter(
                0.1 * torch.randn(d, d)
            ),
            "W_v": nn.Parameter(
                0.1 * torch.randn(d, d)
            ),
            "W_o": nn.Parameter(
                0.1 * torch.randn(d, d)
            ),

            "W_gate": nn.Parameter(
                0.1 * torch.randn(d, d_ff)
            ),
            "W_up": nn.Parameter(
                0.1 * torch.randn(d, d_ff)
            ),
            "W_down": nn.Parameter(
                0.1 * torch.randn(d_ff, d)
            ),
        })

    def forward(self, h):
        # 调用前面实现过的完整 block 计算。
        out, _ = transformer_block(
            h,
            self.params,
            self.num_heads,
        )

        return out


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d,
        d_ff,
        num_heads,
        num_layers,
    ):
        super().__init__()

        # 1. token ID → 向量。
        self.embedding = nn.Embedding(vocab_size, d)

        # 2. 保存多个独立的 Transformer block。
        self.blocks = nn.ModuleList()

        for layer_index in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    d=d,
                    d_ff=d_ff,
                    num_heads=num_heads,
                )
            )

        # 3. 所有 block 之后的最终 RMSNorm 参数。
        self.final_norm_weight = nn.Parameter(
            torch.ones(d)
        )

        # 4. 隐藏向量 → 词表分数。
        self.W_vocab = nn.Parameter(
            0.1 * torch.randn(d, vocab_size)
        )

    def forward(self, input_ids):
        # input_ids：[B,T]

        h = self.embedding(input_ids)
        # h：[B,T,d]

        for block in self.blocks:
            h = block(h)
        # h：[B,T,d]

        h = rms_norm(h, self.final_norm_weight)
        # h：[B,T,d]

        logits = h @ self.W_vocab
        # logits：[B,T,vocab_size]

        return logits


def generate_greedy(
    model,
    input_ids,
    max_new_tokens,
    eos_token_id=None,
):
    """
    单条序列的贪心生成。

    input_ids: [1, T]，至少包含一个 token
    max_new_tokens: 最多新增多少个 token
    eos_token_id: 结束标记 ID；None 表示不检查结束标记

    返回：
    包含原提示和新增 token 的完整 ID 序列。
    """

    # 本节一次生成一条序列。
    assert input_ids.shape[0] == 1

    # 切换到评估模式。
    model.eval()

    # 当前上下文，最初就是输入提示。
    ids = input_ids

    # 生成过程不记录反向计算图。
    with torch.no_grad():
        for step in range(max_new_tokens):

            # 1. 对当前完整上下文做前向计算。
            logits = model(ids)
            # [1, 当前长度, 词表大小]

            # 2. 取最后一个位置的预测分数。
            last_logits = logits[:, -1, :]
            # [1, 词表大小]

            # 3. 选择分数最高的 token ID。
            next_id = last_logits.argmax(
                dim=-1,
                keepdim=True,
            )
            # [1, 1]

            # 4. 把新 token 追加到上下文。
            ids = torch.cat(
                (ids, next_id),
                dim=1,
            )
            # [1, 当前长度 + 1]

            # 5. 如果生成了结束标记，就停止。
            if eos_token_id is not None:
                if next_id[0, 0].item() == eos_token_id:
                    break

    return ids
