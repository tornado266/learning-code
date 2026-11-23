from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=4
        )
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        h = self.embedding(x)
        h = h.mean(dim=1)
        logits = self.linear(h)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    train_sentences = [
        "电影 很 精彩",
        "故事 很 精彩",
        "电影 非常 好看",
        "故事 非常 好看",
        "电影 很 糟糕",
        "故事 很 糟糕",
        "电影 非常 无聊",
        "故事 非常 无聊"
    ]

    val_sentences = [
        "电影 很 好看",
        "故事 非常 精彩",
        "电影 非常 糟糕",
        "故事 很 无聊"
    ]

    train_y = torch.tensor([
        [1.0],  # 电影 很 精彩
        [1.0],  # 故事 很 精彩
        [1.0],  # 电影 非常 好看
        [1.0],  # 故事 非常 好看
        [0.0],  # 电影 很 糟糕
        [0.0],  # 故事 很 糟糕
        [0.0],  # 电影 非常 无聊
        [0.0]   # 故事 非常 无聊
    ], dtype=torch.float32)

    val_y = torch.tensor([
        [1.0],  # 电影 很 好看
        [1.0],  # 故事 非常 精彩
        [0.0],  # 电影 非常 糟糕
        [0.0]   # 故事 很 无聊
    ], dtype=torch.float32)

    vocab = {
        "电影": 0,
        "很": 1,
        "精彩": 2,
        "故事": 3,
        "非常": 4,
        "好看": 5,
        "糟糕": 6,
        "无聊": 7
    }

    train_ids = []

    for sentence in train_sentences:
        sentence_ids = []

        for token in sentence.split():
            token_id = vocab[token]
            sentence_ids.append(token_id)

        train_ids.append(sentence_ids)

    train_x = torch.tensor(train_ids, dtype=torch.long)

    embedding_model = TokenEmbedding(len(vocab))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    embedding_optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.01)

    embedding_model.train()

    for step in range(500):
        embedding_optimizer.zero_grad()

        logits = embedding_model(train_x)

        loss = loss_fn(logits, train_y)

        loss.backward()

        embedding_optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"更新次数:{step + 1},损失:{loss.item():.6f}")

    val_ids = []

    for sentence in val_sentences:
        sentence_ids = []

        for token in sentence.split():
            token_ids = vocab[token]
            sentence_ids.append(token_ids)

        val_ids.append(sentence_ids)

    val_x = torch.tensor(val_ids, dtype=torch.long)

    embedding_model.eval()

    with torch.no_grad():

        logits = embedding_model(val_x)
        val_loss = loss_fn(logits, val_y)

        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities >= 0.5).int()

        correct = predicted_labels == val_y
        accuracy = correct.float().mean().item()

    print("验证损失:", val_loss.item())
    print("好评概率：\n", probabilities)
    print("预测类别：\n", predicted_labels)
    print("正确类别：\n", val_y)
    print(f"验证准确率：{accuracy:.1%}")

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "sentence_classifier.pth"

        torch.save(
            {
                "model_state_dict": embedding_model.state_dict(),
                "vocab": vocab,
            },
            save_path
        )




        # 2. 读取文件
        checkpoint = torch.load(
            save_path,
            map_location="cpu",
            weights_only=True
        )

        # 当前模型类通过 vocab 确定词表大小，所以先恢复词表
        vocab = checkpoint["vocab"]


        # 3. 创建同结构的新模型，再加载训练好的参数
        loaded_model = TokenEmbedding(len(vocab))
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()


        # 4. 用同一批输入，比较原模型和加载后的模型
        embedding_model.eval()

        with torch.no_grad():
            original_probabilities = torch.sigmoid(
                embedding_model(val_x)
            )

            loaded_probabilities = torch.sigmoid(
                loaded_model(val_x)
            )

            original_labels = (original_probabilities >= 0.5).int()
            loaded_labels = (loaded_probabilities >= 0.5).int()

        print("\n加载后的好评概率：")
        print(loaded_probabilities)

        print("\n加载后的预测类别：")
        print(loaded_labels)

        print(
            "\n两个模型的概率是否一致：",
            torch.allclose(original_probabilities, loaded_probabilities)
        )

        print(
            "两个模型的预测类别是否一致：",
            torch.equal(original_labels, loaded_labels)
        )
        assert torch.allclose(original_probabilities, loaded_probabilities)
        assert torch.equal(original_labels, loaded_labels)
