# 2021LeetCode记录

## 简单

### 数组

#### 公平的糖果棒交换：Fair Candy Swap

- 题目描述：爱丽丝和鲍勃有不同大小的糖果棒：A[i] 是爱丽丝拥有的第 i 根糖果棒的大小，B[j] 是鲍勃拥有的第 j 根糖果棒的大小。他们想交换一根糖果棒，交换后他们都有相同的糖果总量。（一个人拥有的糖果总量是他们拥有的糖果棒大小的总和。）返回一个整数数组 ans，其中 ans[0] 是爱丽丝必须交换的糖果棒的大小，ans[1] 是 Bob 必须交换的糖果棒的大小。如果有多个答案，你可以返回其中任何一个。保证答案存在。
- 示例：
输入：A = [1,2,5], B = [2,4]
输出：[5,4]
- 思路：计算A、B的差，此偏差一定为偶数，如偏差为2，则B给出的需要比A给出的大1。此外以set存储可以加快查找。时间复杂度：O（N），空间复杂度：O（N）
- 代码：
```python
def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
    # A和B的偏差，即需要交换的多少，如偏差为2，则A给出的需比B给出的多1个，注意偏差一定为偶数
    bias = (sum(A) - sum(B))/2
    # 使用set存储A，set底层为hash，查找时间复杂度O（1）
    set_A = set(A)
    for b in B:
        a = int(b+bias)
        if a in set_A:
            return [a, b]
```

#### 汇总区间：Summary Ranges

- 问题描述：给定一个无重复元素的有序整数数组nums。返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。列表中的每个区间范围 [a,b] 应该按如下格式输出："a->b" ，如果 a != b；"a" ，如果 a == b
- 示例：
输入：nums = [0,1,2,4,5,7]
输出：["0->2","4->5","7"]
解释：区间范围是：
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"
输入：nums = [-1]
输出：["-1"]
- 思路：一次遍历，时间复杂度:O（N），空间复杂度：O（1）
- 代码：
```python
def summaryRanges(self, nums: List[int]) -> List[str]:
    if len(nums) == 1:
        return [str(nums[0])]
    nums = nums + [0]           # 哨兵
    result = []
    start = 0
    for i in range(len(nums)-1):
        if nums[i]+1 == nums[i+1]:
            continue
        else:
            if start == i:
                result.append(str(nums[i]))
            else:
                result.append(str(nums[start]) + "->" + str(nums[i]))
            start = i + 1 
    return result
```

#### 最长连续递增序列：Longest Continuous Increasing Subsequence

- 题目说明：给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。
- 示例：
输入：nums = [1,3,5,4,7]
输出：3
- 思路：贪心一次遍历，时间复杂度O(N),空间复杂度O（1）
- 代码：
```python
def findLengthOfLCIS(self, nums: List[int]) -> int:
    if len(nums)==1:
        return 1
    result = 0
    start = 0
    # 注：python的for循环并不会每次调用len函数，len只调用1遍生成range对象
    for i in range(1, len(nums)):
        if nums[i] <= nums[i - 1]:
            start = i
        result = max(result, i - start + 1)
    
    return result
```

#### 种花问题：Can Place Flowers

- 问题描述：长花坛的花不能种植在相邻的地块上。一个整数数组flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种花，1 表示种植了花。另有一个数 n ，能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false。
- 样例
输入：flowerbed = [1,0,0,0,1], n = 1
输出：true
- 思路：贪心遍历即可，注意边界值检查，为此我们引入哨兵。时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    # 尾部添加哨兵值
    flowerbed = flowerbed + [0]
    i,length = 0,len(flowerbed)-1

    while i < length:
        # 如果当前位置有花，则当前位置和下一位置都不能种花
        if flowerbed[i]:
            i += 2
        # 如果下一位置有花，那么需要直接i+3，到下下下位置
        elif flowerbed[i+1]:
            i += 3
        # 当前位置和下一位置都无花，在此处种下，只需要修改n不需要修改数组
        else:
            n -=1
            i += 2
    return n<=0
```

#### 数组形式的整数加法：Add to Array-Form of Integer

- 题目描述：对于非负整数 X 而言，X 的数组形式是每位数字按从左到右的顺序形成的数组。例如，如果 X = 1231，那么其数组形式为 [1,2,3,1]。给定非负整数 X 的数组形式 A，返回整数 X+K 的数组形式。
- 示例：
输入：A = [1,2,0,0], K = 34
输出：[1,2,3,4]
解释：1200 + 34 = 1234
- 思路：转为整数再转为数组，时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def addToArrayForm(self, A: List[int], K: int) -> List[int]:
    return map(int, str(int(''.join(map(str,A)))+K))
```

### 动态规划

#### 斐波那契数列：Fibonacci Number

- 问题描述：输入n，返回对应斐波那契数列值
- 示例：输入2，返回1；输入3，返回2
- 思路：动态规划+不缓存减少空间复杂度。时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def fib(self, n: int) -> int:
    if n <2:
        return n
    p, q, r = 0,0,1
    for i in range(2,n+1):
        p,q = q,r
        r = p+q
    return r 
```
- 注：如果用通项公式可以O（1）的时间复杂度

## 中等

### 动态规划

#### 秋叶收藏集：Autumn Leaves Collection

- 题目描述：字符串 leaves 仅包含小写字符 r 和 y， 其中字符 r 表示一片红叶，字符 y 表示一片黄叶。要将收藏集中树叶的排列调整成「红、黄、红」三部分。每部分树叶数量可以不相等，但均需大于等于 1。每次调整操作，可以将一片红叶替换成黄叶或者将一片黄叶替换成红叶。最少需要多少次调整操作才能将秋叶收藏集调整完毕？
- 示例：
输入：leaves = "rrryyyrryyyrr"
输出：2
解释：调整两次，将中间的两片红叶替换成黄叶，得到 "rrryyyyyyyyrr"
- 思路：动态规划
    - 划分三种模式：r（全红），ry（红+黄），ryr（红+黄+红）
    - 如果第i片为r：
        - $f(i).r = f(i-1).r$
        - $f(i).ry = min(f(i-1).r, f(i-1).ry) + 1$（翻转），
        - $f(i).ryr = min(f(i-1).ry, f(i-1).ryr)$
    - 如果第i片为y：
        - $f(i).r = f(i-1).r + 1$ （翻转）
        - $f(i).ry = min(f(i-1).r, f(i-1).ry)$
        - $f(i).ryr = min(f(i-1).ry, f(i-1).ryr) + 1$（翻转）
    - 无需数组缓存，从小到大遍历处理即可，初始化r=1/0，ry和ryr都为1
- 时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def minimumOperations(self, leaves: str) -> int:
    #设计三种模式进行：r模式全红，ry模式红+黄，ryr模式红+黄+红
    # 初始化r为0/1，ry、ryr为无穷大
    r = 1 if leaves[0] == 'y' else 0
    ry, ryr = float('inf'), float('inf')

    # 从1开始，从小到大遍历处理
    for i in range(1, len(leaves)):
        #如果第i片为r，r模式不变，ry模式取r模式最小/ry模式最小加1（本次翻转），ryr模式取ry和ryr模式最小
        if leaves[i] == 'r':
            r, ry, ryr = r, min(r, ry) + 1, min(ry, ryr)
        
        #如果第i片为y，r模式+1（本次翻转），ry模式取r模式/ry模式最小，ryr模式取ry和ryr模式最小+1（本次翻转）
        else:
            r, ry, ryr = r + 1, min(r, ry), min(ry, ryr)+1
    return ryr
```

### 滑动窗口

#### 最大连续1的个数 III：Max Consecutive Ones III

- 题目描述：给定一个由若干 0 和 1 组成的数组 A，我们最多可以将 K 个值从 0 变成 1 。返回仅包含 1 的最长（连续）子数组的长度。
- 示例：
输入：A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
输出：6
解释： [1,1,1,0,0,1,1,1,1,1,1]，如此翻转，最长的子数组长度为 6。
- 思路：转换为求一个最长连续子数组，该子数组最多有K个0。滑动窗口解决
- 时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
    def longestOnes(self, A: List[int], K: int) -> int:
        left, right = 0, 0 # 双指针，右指针主动移动，左指针不满足条件时被动移动
        zeros, max_l = 0, 0
        for right in range(len(A)):
            zeros += 1-A[right]
            while zeros>K:    # 子数组超过K个0，不满足题意
                zeros -= 1-A[left]
                left += 1   # 左指针右移
            max_l = max(max_l, right - left + 1) # 当前子数组和旧max_l取大者
        return max_l
```

### 链表

#### 反转链表 II：Reverse Linked List II

- 题目描述：给出单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请反转从位置 left 到位置 right 的链表节点，返回反转后的链表。
- 样例：
    输入：head = [1,2,3,4,5], left = 2, right = 4
    输出：[1,4,3,2,5]
    输入：head = [5], left = 1, right = 1
    输出：[5]
- 思路：一次遍历实现翻转left~right中间部分的链表。通过从left位置的结点开始头插法，不断将后面的结点头插到前面来，自动实现了中间部分链表翻转
- 代码：
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
    dummy_node = ListNode(-1)   # 首结点
    dummy_node.next = head
    pre = dummy_node
    for _ in range(left - 1):   # 让pre指针停在需要头插的地方
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):   # 开始头插
        aft = cur.next
        cur.next = aft.next
        aft.next = pre.next
        pre.next = aft
    
    return dummy_node.next
```
