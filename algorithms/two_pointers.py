"""用快慢指针和左右指针原地整理数组，或搜索有序数据。"""


def removeDuplicates(nums):
    """原地去重已排序列表，返回去重后前缀的长度。"""
    if len(nums) == 0:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1


def removeElement(nums, val):
    """原地移除指定值并保持其余元素的顺序，返回有效长度。"""
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow


def moveZeroes(nums):
    """把零移到列表末尾，保留其他元素的相对顺序。"""
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    for i in range(slow, len(nums)):
        nums[i] = 0


def reverseArray(nums):
    left = 0
    right = len(nums) - 1
    while left < right:
        a = nums[left]
        nums[left] = nums[right]
        nums[right] = a
        left += 1
        right -= 1


def isPalindrome(s):
    left = 0
    right = len(s) - 1
    while left < right:
        if not s[left] == s[right]:
            return False
        left += 1
        right -= 1
    return True


def twoSumSorted(nums, target):
    """返回有序列表中和为目标的两个下标，找不到时返回空列表。"""
    left = 0
    right = len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1
    return []
