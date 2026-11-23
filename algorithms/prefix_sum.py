"""给不变的数组和矩阵预先建立前缀表，供之后的闭区间查询使用。"""


class NumArray:

    def __init__(self, nums):
        self.pre = [0] * (len(nums) + 1)
        for i in range(1, len(nums) + 1):
            self.pre[i] = self.pre[i - 1] + nums[i - 1]

    def sumRange(self, left, right):
        sum = self.pre[right + 1] - self.pre[left]
        return sum


class Nummatrix:

    def __init__(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        self.pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self.pre[i][j] = self.pre[i - 1][j] + self.pre[i][j - 1] - self.pre[i - 1][j - 1] + matrix[i - 1][j - 1]

    def sumRegion(self, row1, col1, row2, col2):
        return self.pre[row2 + 1][col2 + 1] - self.pre[row1][col2 + 1] - self.pre[row2 + 1][col1] + self.pre[row1][col1]


def maxSquareSum(matrix, k):
    """使用二维前缀和查询 k × k 正方形的最大元素和。"""
    m = len(matrix)
    n = len(matrix[0])
    arr = Nummatrix(matrix)
    best = arr.sumRegion(0, 0, k - 1, k - 1)
    for i in range(m - k + 1):
        for j in range(n - k + 1):
            current = arr.sumRegion(i, j, i + k - 1, j + k - 1)
            if current > best:
                best = current
    return best


class EvenNumber:
    """对矩形区域的偶数个数做前缀查询；0 和负偶数也计入。"""

    def __init__(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        self.pre = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if matrix[i - 1][j - 1] % 2 == 0:
                    value = 1
                else:
                    value = 0
                self.pre[i][j] = self.pre[i - 1][j] + self.pre[i][j - 1] - self.pre[i - 1][j - 1] + value

    def countEven(self, row1, col1, row2, col2):
        return self.pre[row2 + 1][col2 + 1] - self.pre[row1][col2 + 1] - self.pre[row2 + 1][col1] + self.pre[row1][col1]
