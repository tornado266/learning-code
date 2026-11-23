"""用两条合成 ID 序列运行一次训练更新，再逐个生成 5 个 ID。"""

import torch
import torch.nn.functional as F

from architecture import architecture_report
from model import TinyTransformerLM, generate_greedy


def main():
    torch.manual_seed(0)

    vocab_size = 260
    d = 8
    d_ff = 16
    num_heads = 2
    num_layers = 2

    model = TinyTransformerLM(
        vocab_size=vocab_size,
        d=d,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # 两条合成序列，用来验证模型接口和训练流程。
    batch_ids = torch.tensor(
        [
            [10, 20, 30, 40, 50],
            [60, 70, 80, 90, 100],
        ],
        dtype=torch.long,
    )

    input_ids = batch_ids[:, :-1]
    target_ids = batch_ids[:, 1:]

    logits = model(input_ids)

    print("输入形状：", input_ids.shape)
    print("目标形状：", target_ids.shape)
    print("输出形状：", logits.shape)

    param_count = 0

    for parameter in model.parameters():
        param_count += parameter.numel()

    print("总参数量：", param_count)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    # 1. 清空上一轮梯度。
    optimizer.zero_grad(set_to_none=True)

    # 2. 完整模型前向计算。
    logits = model(input_ids)

    # 3. 计算下一个 token 的交叉熵。
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )

    # 4. 梯度经过整个模型反向传播。
    loss.backward()

    # 5. 更新模型参数。
    optimizer.step()

    print("更新前损失：", loss.item())

    logits_after = model(input_ids)

    loss_after = F.cross_entropy(
        logits_after.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )

    print("更新后损失：", loss_after.item())

    prompt_ids = torch.tensor(
        [[10, 20, 30]],
        dtype=torch.long,
    )

    generated_ids = generate_greedy(
        model=model,
        input_ids=prompt_ids,
        max_new_tokens=5,
    )

    print("原提示：", prompt_ids[0].tolist())
    print("完整结果：", generated_ids[0].tolist())
    print("结果形状：", generated_ids.shape)

    report = architecture_report(
        vocab_size=260,
        d=8,
        d_ff=16,
        num_heads=2,
        num_layers=2,
        B=2,
        T=4,
    )

    for name, value in report.items():
        print(name, "：", value)

    # 核对手算参数量与 PyTorch 中实际注册的参数数量。
    assert report["总参数量"] == param_count
    assert logits.shape == (2, 4, vocab_size)
    assert generated_ids.shape == (1, prompt_ids.shape[1] + 5)


if __name__ == "__main__":
    main()
