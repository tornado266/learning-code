"""在差分数组上修改区间的两个边界，再还原原数组。"""


class diff:
    """为非空数组构造差分表，支持闭区间批量加值和最终还原。"""

    def __init__(self, nums):
        n = len(nums)
        if not n >= 1:
            raise IndexError
        self.diff = [0] * n
        if len(nums) > 0:
            self.diff[0] = nums[0]
        for i in range(1, len(nums)):
            self.diff[i] = nums[i] - nums[i - 1]

    def update(self, left, right, value):
        self.diff[left] += value
        if right + 1 < len(self.diff):
            self.diff[right + 1] -= value

    def result(self):
        nums = [0] * len(self.diff)
        nums[0] = self.diff[0]
        for i in range(1, len(nums)):
            nums[i] = self.diff[i] + nums[i - 1]
        return nums


def corpFlightBookings(bookings, n):
    """用差分表汇总航班闭区间的预订座位数，航班编号从 1 开始。"""
    seat = [0] * n
    arr = diff(seat)
    for i in range(len(bookings)):
        arr.update(bookings[i][0] - 1, bookings[i][1] - 1, bookings[i][2])
    return arr.result()


def carPooling(trips, capacity):
    """按左闭右开区间累加乘客数，检查 0 到 1000 的各站是否超载。"""
    arr = diff([0] * 1001)
    for i in range(len(trips)):
        on = trips[i][1]
        out = trips[i][2]
        people = trips[i][0]
        if not 0 <= on < out <= 1000:
            raise IndexError
        arr.update(on, out - 1, people)
    nums = arr.result()
    for i in range(0, len(nums)):
        if nums[i] > capacity:
            return False
    else:
        return True
