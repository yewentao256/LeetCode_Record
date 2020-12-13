# 2020LeetCode记录

- [2020LeetCode记录](#2020leetcode记录)
  - [简单](#简单)
    - [数组](#数组)
      - [三角形的最大周长：Largest Perimeter Triangle——11/29](#三角形的最大周长largest-perimeter-triangle1129)
      - [存在重复元素：Contains Duplicate——12/13](#存在重复元素contains-duplicate1213)
      - [杨辉三角：Pascal's Triangle——12/13](#杨辉三角pascals-triangle1213)
    - [字符串](#字符串)
      - [上升下降字符串：Increasing Decreasing String——11/25](#上升下降字符串increasing-decreasing-string1125)
      - [罗马字符转整数：Roman to Integer——11/29](#罗马字符转整数roman-to-integer1129)
  - [中等](#中等)
    - [bit级别算法](#bit级别算法)
      - [一定范围内与：Bitwise AND of Numbers Range——2020/9/6](#一定范围内与bitwise-and-of-numbers-range202096)
    - [字符串](#字符串-1)
      - [重构字符串：Reorganize String——11/30](#重构字符串reorganize-string1130)
    - [链表](#链表)
      - [奇偶链表：Odd Even Linked List——11/13](#奇偶链表odd-even-linked-list1113)
      - [链表排序：Sort List——11/18](#链表排序sort-list1118)
    - [数组](#数组-1)
      - [加油站：Gas Station——11/24](#加油站gas-station1124)
  - [困难](#困难)

## 简单

### 数组

#### 三角形的最大周长：Largest Perimeter Triangle——11/29

- 题目说明：输入数组，输出面积不为0的三角形最大周长
- 样例：
```c
输入：[2,1,2]
输出：5
输入：[1,2,1]
输出：0
```
- 思路：倒序排列数组，顺序遍历贪心寻找最大周长
- 时间复杂度：O（NlogN）（来自排序）；空间复杂度：O（logN）（来自排序）
- 代码：
```python
class Solution:
    def largestPerimeter(self, A: List[int]) -> int:
        A.sort(reverse = True)
        for i in range(len(A)-2):
            if A[i] < A[i+1]+A[i+2]:
                return A[i]+A[i+1]+A[i+2]
        return 0
```

#### 存在重复元素：Contains Duplicate——12/13

- 题目说明：输入一个数组，如果数组存在重复元素输出True，否则输出False
- 样例
```c
Input: [1,2,3,1]
Output: true
```
- 思路1：哈希表，空间复杂度O（N），时间复杂度O（N）
- 思路2：比较集合长度，空间复杂度O（N），时间复杂度O（N）
- 代码：
```python
def containsDuplicate(self, nums: List[int]) -> bool:
    return len(nums) != len(set(nums))
```

#### 杨辉三角：Pascal's Triangle——12/13

- 题目说明：给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
- 样例：输入3，输出[[1],[1,1],[1,2,1]]
- 思路：根据杨辉三角定义，在杨辉三角中，每个数是它左上方和右上方的数的和，以此计算即可
- 时间复杂度：O（numRows^2）（注：里面有一个循环，长度约等于numRows），空间复杂度：O（1）（所有构建都直接作用于返回值，不考虑result空间复杂度，而新建row的空间复杂度计为O（1））
- 代码：
```python
def generate(self, numRows: int) -> List[List[int]]:
    if numRows == 0:
        return []
    result = [[1]]
    now_row = 2     # 当前需处理行号
    while(now_row<=numRows):
        last_row = result[-1]
        row = [1]   # 新的一行内容
        for i in range(len(last_row)-1):
            row.append(last_row[i]+last_row[i+1])
        row.append(1)
        result.append(row)
        now_row += 1
    return result
```
- 有趣的思考：如果补0，则发现上一行+上一行移位 = 下一行
例如：
```c
0 1 2 1 + 1 2 1 0 = 1 3 3 1
```


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

#### 罗马字符转整数：Roman to Integer——11/29

- 题目说明：输入罗马字符，转为整数
- 样例：
```c
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4
注意：IC 和 IM 这样的例子并不符合题目要求，49 应该写作 XLIX，999 应该写作 CMXCIX 。
```
- 思路：建立哈希表，遍历字符串
- 时间复杂度：O(N), 空间复杂度：O(1)（或者说是建立哈希表的常数）
- 代码
```python
class Solution:
    def romanToInt(self, s: str) -> int:
        # 生成字典，注意“IM”这种表达是不规范的
        dic = {"I":1, "IV":4, "V":5, "IX":9, "X":10, "XL":40, "L":50, "XC":90, "C":100, "CD":400, "D":500, "CM":900, "M":1000}
        result = 0
        i = 0
        while(i<len(s)):
            if i<len(s)-1 and s[i]+s[i+1] in dic:       # 短路运算，如果i为最后一个不会计入此
                result += dic[s[i]+s[i+1]]
                i += 2
            elif s[i] in dic:
                result += dic[s[i]]
                i += 1
            else:
                print("error")
                break
        return result
```

## 中等

### bit级别算法

#### 一定范围内与：Bitwise AND of Numbers Range——2020/9/6

- 题目说明：输入范围[m,n]，0<=m<=n<=2147483647，返回在这个范围中所有数的按位与，inclusive
- 样例：
```c
输入: [5,7]
输出: 4
```
- 思路：暴力与+如果m与n不同阶结果一定为0（如111和1000） → 超时；
- 思路2：观察，发现最终结果一定为m和n的公共前缀，如1100与1111， 最终结果一定是1100，用移位法解决问题
- 代码：
```python
def rangeBitwiseAnd(self, m: int, n: int) -> int:
        t = 0
        while m<n:
            m = m >> 1
            n = n >> 1
            t +=1
        return n<< t
```

### 字符串

#### 重构字符串：Reorganize String——11/30

- 题目说明：输入字符串，重新排布字母，使得相邻两字符不同。如果不能排布返回空串
- 样例：
```c
输入: S = "aab"
输出: "aba"
```
- 思路：统计计数并排序 → 从最大的计数字符开始，逐二放入最大计数的字符，剩余字符填充
- 时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
class Solution:
    def reorganizeString(self, S: str) -> str:
        if len(S) < 2:
            return S
        length = len(S)
        counts = collections.Counter(S)
        maxCount = max(counts.items(), key=lambda x: x[1])[1]   # 返回最大value的键值对元组，所以用[1]取到最大值
        # 如果n偶数，那么达到n/2+1就不可能匹配，如4个中的3个，代码表示为max>(length+1)//2
        # 如果n奇数，那么达到(n+1)/2+1就不可能匹配，如5个中的4个，代码表示为max>(length+1)//2
        if maxCount > (length + 1) // 2:
            return ""
        lst = [""] * length
        even_index, old_index = 0, 1
        halfLength = length // 2
        # 整体而言，先逐二放最多的，再依次填充
        for c, count in counts.items():
            # 只要count<=总长度的一半，那么就可以放在奇数下标，否则必须放在偶数下标
            while count > 0 and count <= halfLength and old_index < length:
                lst[old_index] = c
                count -= 1
                old_index += 2
            while count > 0:
                lst[even_index] = c
                count -= 1
                even_index += 2
        return "".join(lst)
```

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

#### 链表排序：Sort List——11/18

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

### 数组

#### 加油站：Gas Station——11/24

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


