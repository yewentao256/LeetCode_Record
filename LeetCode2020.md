# 2020LeetCode记录

[TOC]

## 简单

### 字符串

#### 上升下降字符串：Increasing Decreasing String——11/25

- 题目说明：输入一个字串，输出一个上升下降字符串
- 样例
```c
输入：s = "aaaabbbbcccc"
输出："abccbaabccba"
解释：正扫一遍，提取升序字符串'abc'，result = "abc"
反扫一遍，提取降序字符串'cba'，result = "abccba"，以此类推
```
- 思路：有序字典存储字符与值，每轮直接把所有还有值的key拿出来加入result。此法操作后所有key值减1
- 空间复杂度：O(N)，时间复杂度：O(N)
- 代码：
```python
class Solution:
    def sortString(self, s: str) -> str:
        dic = dict(sorted(collections.Counter(s)))      # 生成记录数据的字典（有序）
        result = ''
        while(dic):
            result += ''.join(list(dic.keys()))         # 正向字符
            for c in list(dic.keys()):                  # 清理字典
                dic[c] -= 1
                if dic[c]==0:
                    del dic[c]
            result += ''.join(list(dic.keys())[::-1])   # 反向字符
            for c in list(dic.keys()):                   # 清理字典
                dic[c] -= 1
                if dic[c]==0:
                    del dic[c]
        return result
```

## 中等

### 链表

#### 奇偶链表：Odd Even Linked List——11/13

- 题目说明：输入一个链表，奇数位和偶数位分离
- 样例：
```c
Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
```
- 思路：创建三个指针，odd指向奇数位链表，even指向偶数位链表，even_head记录偶数位链表开头，一次遍历即可。
- 空间复杂度：O（1）， 时间复杂度：O（N）
- 代码：
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        # 没有或者只有一个元素，直接返回
        if not head or not head.next:
            return head
        # 指针初始化
        odd = head
        even_head = odd.next
        even = even_head

        # 遍历——循环展开，一次odd指针和even指针都向后移动两位
        while even and even.next:   # even and even.next一起作为条件，如果even==none，不会执行even.next
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next

        # 遍历完成，将odd接上even_head
        odd.next = even_head
        return head
```

#### 链表排序：Sort List

- 题目说明：链表升序排序，要求时间复杂度O(nlogn)，空间复杂度O(1)
- 样例：
```c
Input: head = [4,2,1,3]
Output: [1,2,3,4]
```
- 思路1：转换列表，使用python自带sort——O(nlogn)，空间复杂度O(n)——最终采用（最快）
- 思路2：自顶向下归并排序，先二分拆分到最小单元，再合并。时间复杂度O(nlogn)，空间复杂度O(n)——次快
- 思路3：自底向上归并排序，从最小单元开始合并。时间复杂度O(nlogn)，空间复杂度O(1)——最慢
- 代码：
```python
# 直接使用python自带sort排序O(nlogn)，比自己实现的自顶向下的O(nlogn)快一些
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        tmp = []
        while head:
            tmp.append(head.val)
            head = head.next
        tmp.sort()
        dummy = ListNode(0)
        t = dummy
        for i in tmp:
            ret = ListNode(i)
            t.next = ret
            t = t.next
        return dummy.next
```

### 一次遍历

#### 加油站：Gas Station

- 题目说明：
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
如果题目有解，该答案即为唯一答案。输入数组均为非空数组，且长度相同。输入数组中的元素均为非负数。
- 样例：
```c
输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]
输出: 3
解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
```
- 思路：直觉为双层循环暴力破解。我们要想办法剪枝做到一次遍历。
首先我们可以证明一个结论：**x-y（不妨设x<y) 走不了，那么x-y中间任意一个z出发都走不了。**
由前提有：
    1. $\sum_{i=x}^y gas[i]<\sum_{i=x}^y cost[i]$
    2. $\sum_{i=x}^j gas[i]>=\sum_{i=x}^j cost[i]$  (j取x到y的任意一个值)

    所以：
    $\sum_{i=z}^y gas[i] = \sum_{i=x}^y gas[i] - \sum_{i=x}^{z-1} gas[i]$$<\sum_{i=x}^y cost[i]-\sum_{i=x}^{z-1} cost[i] = \sum_{i=z}^y cost[i]$
    证明完毕，有这个结论后我们即可一次遍历实现全过程
- 空间复杂度：O（1）， 时间复杂度：O（N）
- 代码：
```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # total_gas记录可获得的总油量-总油耗， now_gas记录当前油耗情况， start_index记录出发位置
        total_gas, now_gas, start_index = 0, 0, 0
        for i in range(len(gas)):
            total_gas += gas[i] - cost[i]
            now_gas += gas[i] - cost[i]
            if now_gas < 0:             # 油不够开到i站
                now_gas = 0             # now_gas置零，在新位置重新开始计算油耗情况
                start_index = i + 1     # 将起始位置改成i+1
        if total_gas>=0:                # 如果total_gas>0那么一定有一个点可以环行一周
            return start_index          # 此处的index可以满足它开始到结尾的now_gas判断，那么必定为所需答案
        else:
            return -1
```

## 困难


