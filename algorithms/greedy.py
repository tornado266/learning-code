"""可分割物品的背包、从两边移动的盛水容器，以及整数拆分乘积。"""


def fractional_knapsack(wgt, val, cap):
    # 将重量和价值配对，按单位价值从高到低排序
    items = sorted(
        zip(wgt, val),
        key=lambda item: item[1] / item[0],
        reverse=True
    )

    total = 0.0

    for weight, value in items:
        take = min(weight, cap)
        total += take * (value / weight)
        cap -= take

        if cap == 0:
            break

    return total


def max_capacity(height):
    i, j = 0, len(height) - 1
    best = 0

    while i < j:
        area = min(height[i], height[j]) * (j - i)
        best = max(best, area)

        if height[i] <= height[j]:
            i += 1
        else:
            j -= 1

    return best


def max_product(n):
    if n <= 3:
        return n - 1

    product = 1

    while n > 4:
        product *= 3
        n -= 3

    return product * n
