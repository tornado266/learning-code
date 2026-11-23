from pathlib import Path
from tempfile import TemporaryDirectory

import pickle
import re
import regex


# ============================================================
# 1. 普通文本的预分词规则
# ============================================================

# 分支依次处理：
# 英文缩写、字母、数字、符号、空白。
#
# “ ?”表示最多带上一个普通空格。
# \p{L} 表示 Unicode 字母，\p{N} 表示 Unicode 数字。
#
# 注意：
# 这个正则用于切分普通文本。
# 特殊 token 要先通过 split_special 单独切开。
full_pattern = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def pretokenize(text):
    """把普通文本切成若干字符串片段，保留空格、换行等内容。"""
    return regex.findall(full_pattern, text)


def text_to_chunks(text):
    """把每个预分词片段转换成独立的初始字节 ID 列表。"""
    chunks = []

    for piece in pretokenize(text):
        # UTF-8 编码得到 bytes。
        # list(bytes) 得到 0～255 之间的整数列表。
        chunks.append(list(piece.encode("utf-8")))

    return chunks


# ============================================================
# 2. 统计相邻 pair，以及执行一次合并
# ============================================================

def count_pairs(ids):
    """统计一个 ID 序列内部，各相邻 pair 出现的次数。"""
    pair_counts = {}

    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts


def merge(ids, pair, new_id):
    """从左到右，把指定 pair 替换成 new_id，实际替换不重叠。"""
    new_ids = []
    i = 0

    while i < len(ids):
        if (
            i + 1 < len(ids)
            and (ids[i], ids[i + 1]) == pair
        ):
            new_ids.append(new_id)
            i += 2  # 已经消费两个旧 token。
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids


# 注意：
# count_pairs([97, 97, 97]) 会统计出 (97, 97) 出现 2 次。
# 但 merge([97, 97, 97], (97, 97), 256) 得到 [256, 97]。
#
# 原因：统计相邻关系时允许位置重叠，实际合并时不能重复消费 token。


def count_chunk_pairs(chunks):
    """汇总各 chunk 内部的 pair 频次，不统计跨 chunk 的相邻关系。"""
    total_counts = {}

    for chunk in chunks:
        counts = count_pairs(chunk)

        for pair, count in counts.items():
            total_counts[pair] = total_counts.get(pair, 0) + count

    return total_counts


def merge_chunks(chunks, pair, new_id):
    """把同一条合并规则分别应用到每个 chunk，不跨边界合并。"""
    new_chunks = []

    for chunk in chunks:
        new_chunk = merge(chunk, pair, new_id)
        new_chunks.append(new_chunk)

    return new_chunks


# ============================================================
# 3. 切开特殊 token
# ============================================================

def split_special(text, special_tokens):
    """切开特殊标记，同时把标记本身保留在返回列表中。"""
    if not special_tokens:
        return [text]

    # 长标记优先，避免 <A> 抢先匹配 <A><B> 的前半部分。
    # 这里只调整匹配顺序，不改变特殊 token 的编号顺序。
    ordered_special = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    escaped_specials = []

    for special in ordered_special:
        # 特殊 token 中可能有正则元字符，需要转义。
        escaped_specials.append(re.escape(special))

    body = "|".join(escaped_specials)

    # 捕获括号使 re.split 在结果里保留特殊标记。
    pattern = "(" + body + ")"

    return re.split(pattern, text)


# ============================================================
# 4. BPE 训练
# ============================================================

def train_bpe(text, vocab_size, special_tokens):
    """
    返回：
        vocab：token ID -> bytes
        merges：(左 ID, 右 ID) -> 新 ID
        special_ids：特殊标记字符串 -> token ID

    本教学版约定：
        特殊标记非空、不重复；
        vocab_size 至少容纳 256 个基础字节和所有特殊标记。
    """

    # ① 创建基础词表。
    # ID 0～255 各自表示一个字节。
    vocab = {}

    for token_id in range(256):
        vocab[token_id] = bytes([token_id])

    # 字典的插入顺序保存合并规则的学习顺序。
    merges = {}

    # ② 先登记特殊 token，占用词表位置。
    special_ids = {}

    for special in special_tokens:
        special_id = len(vocab)
        vocab[special_id] = special.encode("utf-8")
        special_ids[special] = special_id

    # ③ 先切开特殊 token，再处理普通文本。
    parts = split_special(text, special_tokens)
    chunks = []

    for part in parts:
        if part not in special_ids:
            chunks.extend(text_to_chunks(part))

    # 特殊 token 不参与普通 BPE 学习。
    # 它两侧的文本也没有拼接到同一个 chunk 中。

    # ④ 计算剩余词表预算。
    # 总词表大小 = 基础字节 + 特殊 token + 学习到的普通 token。
    num_merges = vocab_size - len(vocab)

    for step in range(num_merges):
        # 每轮都根据更新后的 chunks 重新统计频次。
        pair_counts = count_chunk_pairs(chunks)

        # 如果所有 chunk 都没有相邻 pair，提前停止。
        # 因此最终词表可能小于 vocab_size。
        if not pair_counts:
            break

        # 优先选择频次最大的 pair。
        # 频次相同时，依次比较左右 token 的 bytes，取较大的。
        #
        # 比较的是 bytes，不是 token ID，
        # 也不是左右 token 拼接后的 bytes。
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

        # 本实现连续分配 ID，所以 len(vocab) 就是下一个空闲 ID。
        new_id = len(vocab)

        # 新 token 的内容等于左右 token 的字节拼接。
        vocab[new_id] = (
            vocab[best_pair[0]] + vocab[best_pair[1]]
        )

        # 记录这条合并规则。
        merges[best_pair] = new_id

        # 更新全部 chunks。
        # 必须接住返回值，下一轮才能基于新序列统计。
        chunks = merge_chunks(chunks, best_pair, new_id)

    # ⑤ 返回训练结果。
    return vocab, merges, special_ids


# ============================================================
# 5. 函数版编码与解码
# ============================================================

def encode(text, merges):
    """编码普通文本：应用已经学到的规则，不重新训练。"""
    chunks = text_to_chunks(text)

    # 按训练时的学习顺序应用合并规则。
    for pair, new_id in merges.items():
        chunks = merge_chunks(chunks, pair, new_id)

    result = []

    for chunk in chunks:
        result.extend(chunk)

    return result


def encode_with_special(text, merges, special_ids):
    """特殊 token 直接查 ID，普通文本交给 encode。"""
    result = []
    parts = split_special(text, list(special_ids))

    for part in parts:
        if part in special_ids:
            # 特殊 token 对应单个整数 ID，所以用 append。
            result.append(special_ids[part])
        else:
            # 普通文本编码得到 ID 列表，所以用 extend。
            result.extend(encode(part, merges))

    return result


def decode(ids, vocab):
    """先拼接全部 token 的 bytes，再统一进行 UTF-8 解码。"""
    pieces = []

    for token_id in ids:
        pieces.append(vocab[token_id])

    data = b"".join(pieces)

    # 一个 token 可能只是某个汉字的部分字节。
    # 因此不能逐个 token 单独进行 UTF-8 解码。
    return data.decode("utf-8")


# ============================================================
# 6. 封装成 Tokenizer 类
# ============================================================

class Tokenizer:
    """把训练好的配置和编码、解码操作放在一个对象里。"""

    def __init__(self, vocab, merges, special_ids, pattern):
        self.vocab = vocab
        self.merges = merges
        self.special_ids = special_ids
        self.pattern = pattern

    def encode(self, text):
        """
        流程与函数版相同。

        这里把过程写进方法中，显式使用 self.pattern，
        避免对象的编码行为依赖外部全局 full_pattern。
        """
        result = []

        # ① 切开特殊 token。
        parts = split_special(text, list(self.special_ids))

        for part in parts:
            if part in self.special_ids:
                # ② 特殊 token 直接查 ID。
                result.append(self.special_ids[part])

            else:
                # ③ 普通部分使用对象自己的规则预分词。
                pieces = regex.findall(self.pattern, part)

                for piece in pieces:
                    # ④ 每个片段先转换成基础字节 ID。
                    ids = list(piece.encode("utf-8"))

                    # ⑤ 按学习顺序应用已有合并规则。
                    for pair, new_id in self.merges.items():
                        ids = merge(ids, pair, new_id)

                    result.extend(ids)

        return result

    def decode(self, ids):
        return decode(ids, self.vocab)

    def encode_iterable(self, texts):
        """逐段独立编码，再通过 yield 逐个交出 token ID。"""
        for text in texts:
            for token_id in self.encode(text):
                yield token_id

        # 调用本方法得到生成器，开始迭代后才执行。
        # 每次执行到 yield 时交出一个值，然后暂停。
        #
        # 注意：
        # self.encode(text) 仍会先生成当前这一段的 ID 列表。
        # 不会跨输入字符串的边界合并，
        # 所以不保证等于 encode("".join(texts))。


def main():
    with TemporaryDirectory() as directory:
        # ============================================================
        # 7. 完整调用示例
        # ============================================================

        special_tokens = ["<|endoftext|>", "<|pad|>"]
        train_text = "ab ab<|endoftext|> ab!<|pad|>"

        # ① 训练。
        vocab, merges, special_ids = train_bpe(
            train_text,
            vocab_size=260,
            special_tokens=special_tokens,
        )

        # ② 用训练结果创建分词器对象。
        tokenizer = Tokenizer(
            vocab,
            merges,
            special_ids,
            full_pattern,
        )

        # ③ 编码与解码。
        token_ids = tokenizer.encode(train_text)
        restored_text = tokenizer.decode(token_ids)

        print("特殊标记：", special_ids)
        print("词表大小：", len(vocab))
        print("合并规则：", merges)
        print("新 token：", vocab[258], vocab[259])
        print("编码结果：", token_ids)
        print("解码还原：", restored_text == train_text)

        # 对照函数版与对象版。
        function_ids = encode_with_special(
            train_text,
            merges,
            special_ids,
        )
        print("函数版与对象版一致：", function_ids == token_ids)

        # 预期：
        # special_ids = {"<|endoftext|>": 256, "<|pad|>": 257}
        # merges = {(97, 98): 258, (32, 258): 259}
        # vocab[258] = b"ab"
        # vocab[259] = b" ab"
        # token_ids = [258, 259, 256, 259, 33, 257]


        # ============================================================
        # 8. 保存分词器数据
        # ============================================================

        # 保存四项配套数据。
        # 如果重新训练了分词器，要重新构建 state 再保存。
        state = {
            "vocab": tokenizer.vocab,
            "merges": tokenizer.merges,
            "special_ids": tokenizer.special_ids,
            "pattern": tokenizer.pattern,
        }


        state_path = Path(directory) / "tokenizer.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(state, f)


        # ============================================================
        # 9. 加载数据，重新创建对象
        # ============================================================

        # 只加载可信的 pickle 文件：加载过程可以执行代码。
        with open(state_path, "rb") as f:
            loaded = pickle.load(f)

        # 注意读取 loaded，而不是继续使用原来的 state。
        loaded_tokenizer = Tokenizer(
            loaded["vocab"],
            loaded["merges"],
            loaded["special_ids"],
            loaded["pattern"],
        )

        loaded_ids = loaded_tokenizer.encode(train_text)

        print("加载后编码一致：", loaded_ids == token_ids)
        print(
            "加载后解码还原：",
            loaded_tokenizer.decode(loaded_ids) == train_text,
        )

        # 保存的是数据，不是上面的所有函数定义。
        # 重启 Jupyter 内核后，需要先重新运行函数和类的定义，再加载数据。


        # ============================================================
        # 10. 逐段编码示例
        # ============================================================

        texts = ["ab", " ab"]

        # encode_iterable 返回生成器。
        # list 会不断取出生成器中的值，直到结束。
        iterable_ids = list(loaded_tokenizer.encode_iterable(texts))
        print("逐段编码：", iterable_ids)
        # 预期：[258, 259]

        # 输入段之间不会合并：
        print(
            "分别编码 a、b：",
            list(loaded_tokenizer.encode_iterable(["a", "b"])),
        )
        # 预期：[97, 98]

        print("整体编码 ab：", loaded_tokenizer.encode("ab"))
        # 预期：[258]
        assert function_ids == token_ids == loaded_ids
        assert loaded_tokenizer.decode(loaded_ids) == train_text
        assert iterable_ids == [258, 259]


if __name__ == "__main__":
    main()
