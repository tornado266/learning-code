"""使用固定长度的底层列表练习扩容、缩容和原地搬移。"""


class MyArrayList:
    """初始容量至少为 1。size 记录有效元素数，data 可以留有空位。"""

    def __init__(self, init_capacity=2):
        self.data = [None] * init_capacity
        self.size = 0

    def resize(self, new_capacity):
        """分配新列表并逐个复制有效元素；新容量至少为当前元素数。"""
        new_data = [None] * new_capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data

    def add_last(self, e):
        cap = len(self.data)
        if self.size == cap:
            self.resize(cap * 2)
        self.data[self.size] = e
        self.size += 1

    def add(self, index, e):
        if not 0 <= index <= self.size:
            raise IndexError('插入位置越界')
        if len(self.data) == self.size:
            self.resize(len(self.data) * 2)
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]
        self.data[index] = e
        self.size += 1

    def remove(self, index):
        """删除并返回指定元素，底层容量保持不变。"""
        if not 0 <= index < self.size:
            raise IndexError('删除位置越界')
        delate = self.data[index]
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        self.data[self.size - 1] = None
        self.size -= 1
        return delate

    def remove_(self, index):
        """删除后若元素数至多为容量的四分之一，将容量减半。"""
        if not 0 <= index < self.size:
            raise IndexError('删除位置越界')
        deleted_val = self.data[index]
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        self.data[self.size - 1] = None
        self.size -= 1
        if len(self.data) > 1 and self.size <= len(self.data) // 4:
            self.resize(len(self.data) // 2)
        return deleted_val

    def get(self, index):
        if not 0 <= index < self.size:
            raise IndexError
        get_val = self.data[index]
        return get_val

    def set(self, index, e):
        if not 0 <= index < self.size:
            raise IndexError
        old_val = self.data[index]
        self.data[index] = e
        return old_val
