"""用一维状态表计算编辑距离和硬币组合数。"""


def edit_distance(s, t):
    m = len(t)

    # 空字符串变成目标的各个前缀
    dp = list(range(m + 1))

    for i in range(1, len(s) + 1):
        leftup = dp[0]  # 先保存旧行的第一个值
        dp[0] = i      # 当前前缀变成空字符串，要删除 i 次

        for j in range(1, m + 1):
            up = dp[j]  # 覆盖之前，备份旧的上方

            if s[i - 1] == t[j - 1]:
                dp[j] = leftup
            else:
                dp[j] = min(
                    dp[j - 1],  # 新的左边
                    up,         # 旧的上方
                    leftup      # 旧的左上角
                ) + 1

            leftup = up  # 留给右边的下一格使用

    return dp[m]


def coin_combinations(coins, target):
    """用不重复的正整数面值组合出非负目标金额，硬币数量不限。"""
    dp = [0] * (target + 1)
    dp[0] = 1

    for coin in coins:                         # 每次增加一种可用面值
        for amount in range(coin, target + 1): # 金额从小到大
            dp[amount] += dp[amount - coin]
    return dp[target]
