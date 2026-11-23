"""力扣 704。对升序数组执行二分查找，未找到时返回 -1。"""

class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1


if __name__ == '__main__':
    solution = Solution()

    nums = [-1, 0, 3, 5, 9, 12]

    print(solution.search(nums, 9))  # 应该输出 4
    print(solution.search(nums, 2))  # 应该输出 -1
