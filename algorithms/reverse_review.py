"""独立重写的双指针反转，原地修改并返回传入的列表。"""


def reverseArray(nums):
    left = 0
    right = len(nums) - 1
    while left < right:
        tem = nums[left]
        nums[left] = nums[right]
        nums[right] = tem
        left += 1
        right -= 1
    return nums
