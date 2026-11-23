import re

import regex


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
        chunks.append(list(piece.encode("utf-8")))

    return chunks


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


def split_special(text, special_tokens):
    """切开特殊标记，同时把标记本身保留在返回列表中。"""
    if not special_tokens:
        return [text]

    ordered_special = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    escaped_specials = []

    for special in ordered_special:
        escaped_specials.append(re.escape(special))

    body = "|".join(escaped_specials)

    pattern = "(" + body + ")"

    return re.split(pattern, text)


def train_bpe(text, vocab_size, special_tokens):
    """返回训练的词表、合并规则和特殊标记 ID。"""

    vocab = {}

    for token_id in range(256):
        vocab[token_id] = bytes([token_id])

    merges = {}

    special_ids = {}

    for special in special_tokens:
        special_id = len(vocab)
        vocab[special_id] = special.encode("utf-8")
        special_ids[special] = special_id

    parts = split_special(text, special_tokens)
    chunks = []

    for part in parts:
        if part not in special_ids:
            chunks.extend(text_to_chunks(part))

    num_merges = vocab_size - len(vocab)

    for step in range(num_merges):
        pair_counts = count_chunk_pairs(chunks)

        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

        new_id = len(vocab)

        vocab[new_id] = (
            vocab[best_pair[0]] + vocab[best_pair[1]]
        )

        merges[best_pair] = new_id

        chunks = merge_chunks(chunks, best_pair, new_id)

    return vocab, merges, special_ids


def decode(ids, vocab):
    """先拼接全部 token 的 bytes，再统一进行 UTF-8 解码。"""
    pieces = []

    for token_id in ids:
        pieces.append(vocab[token_id])

    data = b"".join(pieces)

    return data.decode("utf-8")


class Tokenizer:
    """把训练好的配置和编码、解码操作放在一个对象里。"""

    def __init__(self, vocab, merges, special_ids, pattern):
        self.vocab = vocab
        self.merges = merges
        self.special_ids = special_ids
        self.pattern = pattern

    def encode(self, text):
        """先识别特殊标记，再按合并顺序编码各段文本。"""
        result = []

        parts = split_special(text, list(self.special_ids))

        for part in parts:
            if part in self.special_ids:
                result.append(self.special_ids[part])

            else:
                pieces = regex.findall(self.pattern, part)

                for piece in pieces:
                    ids = list(piece.encode("utf-8"))

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


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    train_text = "ab ab<|endoftext|> ab!<|pad|>"
    vocab, merges, special_ids = train_bpe(train_text, 260, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_ids, full_pattern)

    for text in [train_text, "ab  ab!\n你好。"]:
        token_ids = tokenizer.encode(text)
        assert tokenizer.decode(token_ids) == text
        print(repr(text), token_ids)
    assert tokenizer.encode("<|pad|>") == [special_ids["<|pad|>"]]
    print("Round-trip checks passed.")
