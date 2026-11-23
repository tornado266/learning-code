"""通过移动窗口边界更新区间和或字符计数。"""


def maxWindowSum(nums, k):
    """求长度为 k 的连续区间最大和，其中 1 <= k <= len(nums)。"""
    window_sum = 0
    for i in range(0, k):
        window_sum += nums[i]
    best = window_sum
    for right in range(k, len(nums)):
        window_sum = window_sum + nums[right] - nums[right - k]
        if window_sum > best:
            best = window_sum
    return best


def minSubArrayLen(target, nums):
    """正整数数组中求和至少为正数 target 的最短连续区间长度，无解返回 0。"""
    left = 0
    window_sum = 0
    best = len(nums) + 1
    for right in range(len(nums)):
        window_sum += nums[right]
        while window_sum >= target:
            best = min(best, right - left + 1)
            window_sum -= nums[left]
            left += 1
    if best == len(nums) + 1:
        return 0
    else:
        return best


def longestOnes(nums, k):
    """最多允许把 k 个 0 看作 1，求最长连续 1 的个数。"""
    left = 0
    zeros = 0
    best = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


def lengthOfLongestSubstring(s):
    left = 0
    best = 0
    counts = {}
    for right in range(len(s)):
        c = s[right]
        counts[c] = counts.get(c, 0) + 1
        while counts[c] > 1:
            counts[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


def totalFruit(fruits):
    """求至多包含两种不同值的最长连续区间。"""
    left = 0
    counts = {}
    best = 0
    for right in range(len(fruits)):
        c = fruits[right]
        counts[c] = counts.get(c, 0) + 1
        while len(counts) > 2:
            d = fruits[left]
            counts[d] -= 1
            if counts[d] == 0:
                del counts[d]
            left += 1
        best = max(best, right - left + 1)
    return best


def checkInclusion(s1, s2):
    """判断 s2 中是否有 s1 的排列组成的连续子串。"""
    if not s1:
        return True
    count = {}
    window = {}
    left = 0
    k = len(s1)
    for i in range(len(s1)):
        c = s1[i]
        count[c] = count.get(c, 0) + 1
    for right in range(len(s2)):
        c = s2[right]
        window[c] = window.get(c, 0) + 1
        if right - left + 1 > k:
            d = s2[left]
            window[d] -= 1
            if window[d] == 0:
                del window[d]
            left += 1
        if right - left + 1 == k and window == count:
            return True
    return False


def cover(count, window):
    """检查当前字符计数 count 是否覆盖需求表 window。"""
    for c in window:
        if count.get(c, 0) < window.get(c, 0):
            return False
    return True


def minWindow(s, t):
    """返回覆盖目标字符及其数量的最短子串；无解或目标为空时返回空串。"""
    if not t:
        return ''
    window = {}
    count = {}
    left = 0
    best_len = len(s) + 1
    best_start = 0
    for i in range(len(t)):
        c = t[i]
        window[c] = window.get(c, 0) + 1
    for right in range(len(s)):
        c = s[right]
        count[c] = count.get(c, 0) + 1
        while cover(count, window):
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_start = left
            d = s[left]
            count[d] -= 1
            if count[d] == 0:
                del count[d]
            left += 1
    if best_len == len(s) + 1:
        return ''
    return s[best_start:best_start + best_len]
