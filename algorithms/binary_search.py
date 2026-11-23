"""在有序数组中查找边界，再对一组可行答案进行二分。"""


def binarySearch(nums, target):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (right + left) // 2
        if nums[mid] == target:
            return mid
        if target > nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def firstPosition(nums, target):
    left = 0
    right = len(nums) - 1
    answer = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            answer = mid
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
            pass
        else:
            right = mid - 1
            pass
    return answer


def lastPosition(nums, target):
    left = 0
    right = len(nums) - 1
    answer = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            answer = mid
            left = mid + 1
        elif nums[mid] < target:
            left = mid + 1
            pass
        else:
            right = mid - 1
            pass
    return answer


def countTarget(nums, target):
    left = firstPosition(nums, target)
    right = lastPosition(nums, target)
    if left == right == -1:
        return 0
    return right - left + 1


def searchInsert(nums, target):
    """在升序无重复数组中求目标下标或插入位置。"""
    left = 0
    right = len(nums) - 1
    answer = binarySearch(nums, target)
    if answer != -1:
        return answer
    while left <= right:
        mid = (right + left) // 2
        if target > nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return left


def mySqrt(x):
    """返回非负整数的平方根向下取整。"""
    left = 0
    right = x
    answer = 0
    while left <= right:
        mid = (left + right) // 2
        if mid ** 2 <= x:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    return answer


def minEatingSpeed(piles, h):
    """对速度做二分，求 h 小时内吃完所有非空香蕉堆所需的最小整数速度。"""
    left = 1
    right = max(piles)
    answer = right
    while left <= right:
        k = (left + right) // 2
        hours = 0
        for pile in piles:
            hours += pile // k
            if pile % k != 0:
                hours += 1
        if hours <= h:
            answer = k
            right = k - 1
        else:
            left = k + 1
    return answer


def shipWithinDays(weights, days):
    """对载重做二分，求按原顺序在给定天数内运完全部包裹的最小载重。"""
    left = max(weights)
    right = sum(weights)
    answer = right
    while left <= right:
        mid = (left + right) // 2
        used_days = 1
        load = 0
        for weight in weights:
            if load + weight > mid:
                used_days += 1
                load = 0
            load += weight
        if used_days <= days:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    return answer


def splitArray(nums, k):
    """把非负整数数组分成 k 个非空连续部分，使最大段和最小。"""
    left = max(nums)
    right = sum(nums)
    answer = right
    while left <= right:
        parts = 1
        total = 0
        mid = (left + right) // 2
        for num in nums:
            if total + num > mid:
                parts += 1
                total = 0
            total += num
        if parts <= k:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    return answer
