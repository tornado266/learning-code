# 数据结构与算法练习

一开始我用二分查找找数组里的数。后来做「分割数组」时，循环仍然每次排除一半，查找的对象却变成了每一段允许有多大的和。下面这组 Python 练习记录了从数组基本操作到这类问题的过程。每个文件都可以单独阅读，只用到 Python 标准库。

## 从一道题开始

把 `[7, 2, 5, 10, 8]` 切成两个连续部分，希望较大的那段和尽量小。把上限设为 17 时，需要切成三段；放宽到 18 时，`[7, 2, 5]` 和 `[10, 8]` 两段就够了。因此 18 是最小可行上限。[查看 splitArray 的二分与逐段累加](binary_search.py)

## 按问题找实现

| 主题 | 可以试什么 |
| --- | --- |
| [动态数组](dynamic_array.py) | 比较普通删除 `remove` 和带缩容的 `remove_`。添加、插入和删除时逐个搬移元素，还能检查有效长度与底层容量。 |
| [数组双指针](two_pointers.py) | 用一个指针读、一个指针写，实现有序数组去重、删除指定值、移动零；从两端逼近可以反转数组、检查回文、找两数之和。 |
| [滑动窗口](sliding_window.py) | 从定长窗口最大和到最短覆盖子串，练习增减元素、维护字符数量、判断何时缩小窗口。 |
| [二分查找](binary_search.py) | 查找重复值的左右边界、计数和插入位置，再把同样的思路用于整数平方根、最小速度、船只载重与数组分割。 |
| [前缀和](prefix_sum.py) | 预先累加一次后，常数时间查询数组或矩阵闭区间的和；还可以找最大正方形总和，或统计矩形里的偶数。 |
| [差分数组](difference_array.py) | 每次只更新区间的两个端点，再一次还原全部结果。两个应用分别是汇总航班预订和核对拼车人数。 |
| [反转数组的复习稿](reverse_review.py) | 重写左右指针交换，这一版原地修改后也返回同一个列表。 |
| [力扣 704 提交版](leetcode_704.py) | 保留平台要求的 `Solution.search` 接口，附两个可以直接运行的例子。 |
| [贪心](greedy.py) | 比较分数背包的单位价值排序、盛水容器的短边移动和整数拆分的乘积。 |
| [动态规划](dynamic_programming.py) | 比较滚动数组算编辑距离与逐枚硬币更新组合数。 |

## 逐步观察算法

以下每个链接都会打开对应 HTML 的 GitHub 源文件。使用文件页右上方的 **Download raw file** 保存为 `.html`，再双击用浏览器离线打开。可以逐步切换状态，或拖动规模滑块查看配对数量的变化。

| 主题 | 交互图解 |
| --- | --- |
| 基础结构与复杂度 | [两两配对](visualizations/quadratic-pairs.html) · [数组插入](visualizations/array-insertion.html) · [栈](visualizations/stack-operations.html) · [循环队列](visualizations/circular-queue.html) |
| 查找与排序 | [二分查找](visualizations/binary-search-intro.html) · [堆排序](visualizations/heap-sort-steps.html) · [桶排序](visualizations/bucket-sort-lesson.html) · [计数排序](visualizations/counting-sort-lesson.html) · [基数排序](visualizations/radix-sort-lesson.html) |
| 递归与回溯 | [汉诺塔](visualizations/hanoi-three-disks.html) · [四皇后](visualizations/four-queens-backtrack.html) |
| 动态规划与贪心 | [网格滚动数组](visualizations/grid-rolling-dp.html) · [背包的遍历方向](visualizations/knapsack-direction.html) · [编辑距离](visualizations/edit-distance-rolling.html) · [盛水容器双指针](visualizations/max-capacity-pointers.html) |

## 跑几个边界

在本目录使用 Python 3.9 或更高版本运行 `python -m doctest README.md`，可以复核以下示例。单独运行 `python leetcode_704.py` 会打印一次命中和一次未命中的结果。

```pycon
>>> from binary_search import splitArray
>>> splitArray([7, 2, 5, 10, 8], 2)
18
>>> from sliding_window import maxWindowSum, minWindow
>>> maxWindowSum([-5, -2, -8], 2)
-7
>>> minWindow("ADOBECODEBANC", "ABC")
'BANC'
>>> from difference_array import carPooling
>>> carPooling([[2, 0, 3], [2, 3, 5]], 2)
True

```

