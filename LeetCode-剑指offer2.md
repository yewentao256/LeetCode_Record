# 剑指offer2

## 简单

### 二叉搜索树的第k大节点

- 题目描述：给定一棵二叉搜索树（二叉排序树），请找出其中第k大的节点。

- 代码：倒中序遍历（先访问最大的，再依次访问值逐渐减少的）

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`——最差的链表情况，全部入栈

```py
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:

        self.k, self.k_node_val = k, None

        def recur(node: TreeNode):
            if node is None:
                return
            
            recur(node.right)
            if self.k == 1:
                self.k_node_val = node.val
                self.k -= 1
                return
            self.k -= 1
            recur(node.left)

        recur(root)
        return self.k_node_val
```

### 0～n-1中缺失的数字

- 题目描述：一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

- 示例：

```bash
输入: [0,1,3]
输出: 2
```

- 代码：二分法

- 时间复杂度：`O（logn）`，空间复杂度：`O（1）`

```py
def missingNumber(nums: List[int]) -> int:
    left, right = 0, len(nums) - 1

    # 标准二分，出循环时，right = left - 1，此时left指向答案
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == mid:
            left = mid + 1
        else:
            right = mid - 1
    return left
```

### 左旋转字符串

- 题目描述：字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

- 代码：切片

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def reverseLeftWords(s: str, n: int) -> str:
    # 切片效率最高。列表后join次之。新建字符串逐一拼接效率最低（python中字符串不可变）
    return s[n:] + s[:n]
```

- 备注：列表后join就像旋转数组那样解决即可

### 在排序数组中查找数字 I

- 题目描述：统计一个数字在排序数组中出现的次数。

- 示例：

```bash
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

- 代码：二分查找，找到上下边界，直接计算

时间复杂度：`O（logn）`，空间复杂度：`O（1）`

```py
def search(nums: [int], target: int) -> int:
    # 搜索右边界 high
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    # 注意：出循环时，right = left - 1，一定是target的指向
    high = left
    # 若数组中无 target ，则提前返回
    if right >= 0 and nums[right] != target:
        return 0
    # 搜索左边界 low
    left = 0
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1
    low = right         # right会走到刚好出边界之地

    # 如[5,7,7,8,8,10], low = 2, high = 5
    return high - low - 1
```

- 注：两次二分逻辑差不多，可以简化代码

```py
def search(nums: [int], target: int) -> int:
    def search_boundary(target: int) -> int:
        # 找到目标值+1的第一个元素
        # 如[2, 7, 7, 8]找7，返回8的位置
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    return search_boundary(target) - search_boundary(target - 1)
```

### 翻转单词顺序

- 题目描述：

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

- 示例：

```bash
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

- 代码：遍历

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def reverseWords(s: str) -> str:
    # 注意：如果可以用内置函数，python用split()最佳
    result = []
    word = ''
    for c in s:
        if c == ' ' and len(word) > 0:
            result.append(word)
            word = ''
        
        if c != ' ':
            word += c
    
    if word:
        result.append(word)
    
    return ' '.join(result[::-1])
```

### 两个链表的第一个公共节点

- 题目描述：输入两个链表，找出它们的第一个公共节点。

- 示例：

```bash
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
```

- 代码1：哈希表，遍历A存入哈希表，再遍历B，有的话就相交

- 代码2：双指针，浪漫相遇~

如果没有公共结点，会一起到None。如果有公共结点，那就会一起到公共结点。

时间复杂度：`O（n + m）`，空间复杂度：`O(1)`

```py
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    
    nodeA, nodeB = headA, headB
    
    while nodeA != nodeB:
        # 如果走完，换到另一个节点开始
        nodeA = nodeA.next if nodeA else headB
        nodeB = nodeB.next if nodeB else headA

    return nodeA
```

### 和为s的连续正数序列

- 题目描述：输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

- 示例：

```bash
输入：target = 9
输出：[[2,3,4],[4,5]]
```

- 代码：滑动窗口

时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def findContinuousSequence(target: int) -> List[List[int]]:
    # 法一：暴力枚举，注意只需要前一半就可以了
    # 法二：滑动窗口
    if target <=2:
        return []

    left, right, total = 1, 2, 3
    mid = target // 2 + 1
    result = []
    while right <= mid:

        # 比target大，左指针右移
        if total > target:
            total -= left
            left += 1
        
        # 找到目标
        elif total == target:
            result.append(list(range(left, right+1)))
            total -= left
            left += 1
            right += 1
            total += right
        
        # 比target小，右指针右移
        else:
            right += 1
            total += right
            
    return result
```

### 和为s的两个数字

- 题目描述：输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

- 示例：

```bash
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```

- 代码：双指针，大了右指针向左，小了左指针向右

时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def twoSum(nums: List[int], target: int) -> List[int]:
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s > target:
            right -= 1
        elif s == target:
            return [nums[left], nums[right]]
        else:
            left += 1

```

- 注：可以考虑加入二分查找，降低时间复杂度。具体指先尝试找到中间值target//2，然后从中间值开始双指针向两边延伸。

### 二叉树的深度

- 题目描述：输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

- 代码：层序遍历

时间复杂度：`O（N）`，空间复杂度：`O（k）`，最差情况下（平衡二叉树），存储N/2个结点，O（N）

```py
def maxDepth(root: TreeNode) -> int:
    queue = deque()     # 也可以不用queue，而是用列表，每次重新刷新列表即可
    queue.append(root)
    
    count = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node:
                queue.append(node.left)
                queue.append(node.right)
        count += 1     
    
    return count - 1         # 这样统计，最下一层的None会加入队列，因此多一层
```

### 第一个只出现一次的字符

- 题目描述：在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

- 示例：

```bash
输入：s = "abaccdeff"
输出：'b'
```

- 代码（哈希表）：

时间复杂度：`O（N）`，空间复杂度：`O（1）`（常数个key）

```py
def firstUniqChar(s: str) -> str:
    dic = defaultdict(int)
    for c in s:
        dic[c] += 1
    for c, count in dic.items():
        if count == 1:
            return c
    return ' '
```

### 数组中出现次数超过一半的数字

- 题目描述：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

- 代码1（哈希表）：
时间复杂度和空间复杂度均`为O（N）`，但比投票法更快因为2/3部分差不多就能return

```py
def majorityElement(nums: List[int]) -> int:
    dic = defaultdict(int)
    target = len(nums) / 2

    for num in nums:
        dic[num] += 1
        if dic[num] >= target:      # 用大于等于，数字出现刚好为一半的也能取到
            return num
```

- 代码2（投票法）

空间复杂度降低至`O（1）`，但需要遍历完整个数组，不推荐

```py
def majorityElement(nums: List[int]) -> int:
    votes = 0
    for num in nums:
        if votes == 0:
            x = num
        if num == x:
            votes += 1 
        else:
            votes -= 1
    return x
```

### 连续子数组的最大和

- 题目描述：输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

- 示例

```bash
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

- 代码：动态规划

时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def maxSubArray(nums: List[int]) -> int:
    # dp[i]：以第i个数为结尾的最大子数组和, dp[0] = nums[0]
    dp = [nums[0]] * len(nums)

    # 状态转移：
    # 如果dp[i-1] < 0，说明dp[i-1]负贡献，不要更好：dp[i] = nums[i]
    # 如果dp[i-1] > 0： dp[i] = nums[i] + dp[i-1]
    for i in range(1, len(nums)):
        if dp[i-1] > 0:
            dp[i] = dp[i-1] + nums[i]
        else:
            dp[i] = nums[i]
    
    # 注：此题可以优化空间复杂度，把dp直接存到nums里，因为nums之前的数无关了
    return max(dp)

```

### 包含min函数的栈

- 题目描述：定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 `min` 函数在该栈中，调用 `min`、`push` 及 `pop` 的时间复杂度都是 `O(1)`。

- 示例：

```bash
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   # --> 返回 -3.
minStack.pop();
minStack.top();   # --> 返回 0.
minStack.min();   # --> 返回 -2.
```

- 代码：辅助栈，保证辅助栈的栈顶是最小值即可。（因为min不会修改数组，所以相对简单）

```py
class MinStack:
    def __init__(self):
        self.A, self.B = [], []

    def push(self, x: int) -> None:
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self) -> None:
        if self.A.pop() == self.B[-1]:
            self.B.pop()

    def top(self) -> int:
        return self.A[-1]

    def min(self) -> int:
        return self.B[-1]
```

### 最小的k个数

- 题目描述：输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

- 示例：

```bash
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

- 代码1：排序后切片 `return sorted(arr)[:k]`，时间复杂度：`O（NlogN）`，空间复杂度`O（N）`

- 代码2：利用快排不完全排序，仅仅找到k个最小值。时间复杂度:`O（N）`，空间复杂度：`O（logN）`

时间复杂度解析：
首先快排是因为每次递归操作N次，递归logN次，所以`O(N*logN)`
这里是假定都取平均递归子数组为N/2，那么总操作次数 = N + N/2 + N/4 + ... < 2N
等比数列: `a1 * (1 - q^n)/(1-q)`

```py
def getLeastNumbers(arr: List[int], k: int) -> List[int]:
    if k >= len(arr):
        return arr

    def quick_sort(low: int, high: int) -> List[int]:
        left, right = low, high
        key = arr[low]
        while left < right:
            while left < right and arr[right] >= key:
                right -= 1
            arr[left] = arr[right]
            while left < right and arr[left] <= key:
                left += 1
            arr[right] = arr[left]
        
        # 出循环，left = right
        arr[left] = key

        # 优化：我们不需要完全排好序，我们只需要找对应k个最小值即可
        if k < left:
            return quick_sort(low, left - 1) 
        if k > left:
            return quick_sort(left + 1, high)
        return arr[:k]
        
    return quick_sort(0, len(arr) - 1)
```

### 删除链表的节点

- 题目描述：给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回头结点

- 示例：

```bash
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
```

- 代码：遍历找到然后删除

时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def deleteNode(head: ListNode, val: int) -> ListNode:
    pre, cur = ListNode(-1), head
    pre.next = head
    dummy = pre
    while cur:
        if cur.val == val:
            pre.next = cur.next
            break
        pre, cur = cur, cur.next

    return dummy.next
```

### 反转链表

- 题目描述：定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

- 示例：

```bash
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

- 代码：**双指针一次就地遍历**

时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def reverseList(head: ListNode) -> ListNode:
    cur, pre = head, None
    while cur:
        cur.next, pre, cur = pre, cur, cur.next
    return pre
```

### 两个栈实现队列

- 题目说明：用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

- 代码：

```py
class CQueue:

    def __init__(self):
        self.put_stack = []
        self.get_stack = []

    def appendTail(self, value: int) -> None:
        self.put_stack.append(value)

    def deleteHead(self) -> int:
        if self.get_stack:
            return self.get_stack.pop()
        while self.put_stack:
            self.get_stack.append(self.put_stack.pop())
        # 如果有数据，return 数据，否则return -1
        if self.get_stack:
            return self.get_stack.pop()
        return -1
```

- 时间复杂度：`O（1）`，因为弹出操作对每个元素最多只操作一次，因此均摊下来为`O（1）`；空间复杂度：`O（N）`

### 斐波那契数列

- 题目说明：0 1 1 2 3 5 8 斐波那契数列求解

#### 代码1：简易动态规划

```py
def fib(n: int) -> int:
    MOD = 10 ** 9 + 7   # LeetCode评测要求取模
    if n < 2:
        return n
    p, q, r = 0, 0, 1
    for i in range(2, n + 1):
        p = q
        q = r
        r = p + q
    return r % MOD
```

时间复杂度：`O（N）`，空间复杂度：`O（1）`

#### 代码2：矩阵快速幂

矩阵快速幂的基本原理：

\[ a^{b} = a ^{b1+b2} = a^{b1} * a^{b2} \]

我们从指数最低位开始处理，每次处理一位，实际上是将指数翻倍（对应底数平方），`b2 = 2*b1`

矩阵快速幂的代码：

```py
def fastPow(a, b):
    result = 1
    while b > 0:
        if b & 1:  # 如果b的当前最低位是1
            result *= a
        a *= a  # 底数平方
        b >>= 1  # b右移一位
    return result

# 例如：a=3（底数） b=5（指数，二进制101）
# iter1：result = 1 * 3 = 3，a平方变为9，b右移变为2（10）
# iter2：不更新result（无贡献），a平方变为81，b右移变为1
# iter3：更新result = 3 * 81 = 243，结束
```

所以我们只需关心二进制位有贡献的权重累乘即可，将乘法操作数从n次降低到了log次

回到本题，我们可以得到斐波那契数列有：

\[ \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} F(n) \\ F(n-1) \end{pmatrix} \ = \begin{pmatrix} F(n) + F(n-1) \\ F(n) \end{pmatrix} = \begin{pmatrix} F(n+1) \\ F(n) \end{pmatrix} \]

进而表示为：

\[ \begin{pmatrix} F(n) \\ F(n-1) \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^{n-1} \begin{pmatrix} F(1) \\ F(0) \end{pmatrix} \]

```py
class Solution:
    def fib(self, n: int) -> int:
        MOD = 10 ** 9 + 7   # MOD为LeetCode评测要求
        if n < 2:
            return n
        
        # 矩阵乘（二维）
        def multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
            c = [[0, 0], [0, 0]]
            for i in range(2):
                for j in range(2):
                    c[i][j] = (a[i][0] * b[0][j] + a[i][1] * b[1][j]) % MOD
            return c

        # 矩阵快速幂核心算法
        def matrix_pow(a: List[List[int]], n: int) -> List[List[int]]:
            ret = [[1, 0], [0, 1]]
            while n > 0:
                if n & 1:
                    ret = multiply(ret, a)
                n >>= 1
                a = multiply(a, a)
            return ret

        res = matrix_pow([[1, 1], [1, 0]], n - 1)
        return res[0][0]
```

### 数组中重复的数字

- 题目说明：找出数组中重复的数字。

在一个长度为 `n` 的数组 `nums` 里的所有数字都在 `0～n-1` 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

- 示例：

```bash
输入： "[2, 3, 1, 0, 2, 5, 3]"
输出：2 或 3
```

- 代码：遍历放入set，看是否增加长度

```py
def findRepeatNumber(nums: List[int]) -> int:
    s = set()
    l = 0
    for num in nums:
        s.add(num)
        l += 1
        if len(s) != l:
            return num
```

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`（会比dict稍微好一点）

### 青蛙跳台阶

- 题目说明：一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 `n` 级的台阶总共有多少种跳法。答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

- 示例：

```bash
输入：n = 0
输出：1
```

- 代码：一维动态规划，设定`lst[i]`为跳上第n个台阶的跳法，`lst[0]=1, lst[1]=1`，状态转移方程`lst[i] = lst[i-1] + lst[i-2]`

```py
def numWays(n: int) -> int:
    MOD = 10 ** 9 + 7
    lst = [1, 1]
    for i in range(2, n+1):
        new_ele = (lst[i-1]+lst[i-2]) % MOD
        lst.append(new_ele)
    return lst[n]
```

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`
- 注：也可以使用循环变量法，`a, b = b, a + b`，使得空间复杂度降至`O（1）`

### 旋转数组的最小数字

- 题目说明：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 `[3,4,5,1,2]` 为 `[1,2,3,4,5]` 的一个旋转，该数组的最小值为`1`。  

- 示例：

```bash
输入：[3,4,5,1,2]
输出：1
```

- 代码：二分查找

时间复杂度：`O（logN）`，空间复杂度：`O（1）`

```py
def minArray(numbers: List[int]) -> int:
    left, right = 0, len(numbers) - 1
    while left < right:
        p = (left+right) // 2
        if numbers[p] > numbers[right]:     # 中间值比最右值大，最小值位于中间值右侧
            left = p + 1
        elif numbers[p] < numbers[right]:   # 中间值比最右值小，最小值位于中间值左侧
            right = p
        else:                               # 中间值恰为最右值，最右值往左移动
            right -= 1
    
    return numbers[left]
```

### 替换空格

- 题目描述：请实现一个函数，把字符串 `s` 中的每个空格替换成`"%20"`。

- 示例：略

- 代码：

```py
def replaceSpace(s: str) -> str:
    return '%20'.join(s.split(' '))
```

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`
- 注：可以通过其他方式比如扩充字串呀什么的实现O（1）空间复杂，但没有必要

### 从头到尾打印列表

- 题目说明：输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

- 示例：略

- 代码：加入列表后倒序

```py
def reversePrint(head: ListNode) -> List[int]:
    stack = []
    while head:
        stack.append(head.val)
        head = head.next
    return stack[::-1]
```

- 时间复杂度：`O(N)`, 空间复杂度：`O（N）`

### 合并两个排序的链表

- 题目说明：输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

- 示例：

```bash
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

- 代码：遍历比大小加入即可

```py
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:

    node = head = ListNode(0)
    
    while l1 and l2:
        if l1.val <= l2.val:
            node.next, l1 = l1, l1.next
        else:
            node.next, l2 = l2, l2.next
        node = node.next

    # 此时l1或l2已经为空，那么直接把剩下的续在结点后即可
    node.next = l1 if l1 else l2    
    return head.next
```

- 时间复杂度：`O（N）`或者说 `O（N1+N2）`，空间复杂度：`O（1）`

### 二叉树的镜像

- 题目说明：请完成一个函数，输入一个二叉树，该函数输出它的镜像。

- 示例：

```bash
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

- 代码：递归交换

```py
def mirrorTree(root: TreeNode) -> TreeNode:
    if not root:
        return
    root.left, root.right = root.right, root.left
    
    mirrorTree(root.left)
    mirrorTree(root.right)

    return root
```

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`（栈深。最坏情况下，二叉树退化为链表）

### 对称的二叉树

- 题目说明：请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

- 示例：

```bash
输入：root = [1,2,2,3,4,4,3]
输出：true
输入：root = [1,2,2,null,3,null,3]
输出：false
```

- 代码：递归比较

```py
def isSymmetric(root: TreeNode) -> bool:
    def recur(L, R):
        if not L and not R:
            return True
        if not L or not R or L.val != R.val:
            return False
        return recur(L.left, R.right) and recur(L.right, R.left)

    return recur(root.left, root.right) if root else True
```

- 时间复杂度：`O（N）`，空间复杂度：`O（N）`（栈深，最差情况下，三角延伸）

### 调整数组顺序使奇数位于偶数前面

- 题目说明：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

- 示例：

```bash
输入：nums = [1,2,3,4]
输出：[1,3,2,4]
注：[3,1,2,4] 也是正确的答案之一。
```

- 代码：双指针，左指针从左到右，右指针从右到左，找到后交换

```py
def exchange(nums: List[int]) -> List[int]:
    left, right = 0, len(nums)-1

    while left < right:
        # 找到左指针指向为奇数的
        while left < right and nums[left] &1 == 1:    # 按位与& 比%要快挺多
            left += 1
        # 找到右指针指向为奇数的
        while left < right and nums[right] &1 ==0:
            right -= 1

        nums[left], nums[right] = nums[right], nums[left]
        left +=1
    return nums
```

- 时间复杂度：`O（N）`，空间复杂度：`O（1）`

### 二进制中1的个数

- 题目说明：编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 `'1'` 的个数（也被称为 汉明重量）。

- 示例：
输入：n = 11 (控制台输入 00000000000000000000000000001011)
输出：3

- 代码：

```py
def hammingWeight(n: int) -> int:
    res = 0
    while n:
        res += n & 1
        n >>= 1
    return res
```

- 时间复杂度：`O（$log_2{n}$）`，空间复杂度：`O（1）`

### 顺时针打印矩阵

- 题目描述：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

- 示例：

```bash
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

- 代码：顺序打印，时间复杂度：`O（M * N）`，空间复杂度：`O（1）`

```py
def spiralOrder(matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []

    # 可以访问的边界，左、上、右、下
    left, up, right, down = 0, 0, len(matrix[0])-1, len(matrix)-1
    result = []
    while True:
        # 左到右，上界限 + 1
        for i in range(left, right + 1):
            result.append(matrix[up][i])
        up += 1
        if up>down:
            break

        # 上到下，右界限 + 1
        for j in range(up, down + 1):
            result.append(matrix[j][right])
        right -= 1
        if left > right:
            break

        # 右到左，下界限 + 1
        for i in range(right, left - 1, -1):
            result.append(matrix[down][i])
        down -= 1
        if up > down:
            break

        # 下到上，左界限 + 1
        for j in range(down, up-1, -1):
            result.append(matrix[j][left])
        left += 1
        if left > right:
            break
    return result
```

### 链表中倒数第k个节点

- 题目描述：输入一个链表，输出该链表中倒数第k个节点。

- 示例：

```bash
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
```

- 代码：双指针，右指针先走两步，时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def getKthFromEnd(head: ListNode, k: int) -> ListNode:
    right = left = head
    
    for i in range(k):
        right = right.next
    
    while right:
        left = left.next
        right = right.next
    
    return left
```

### 打印从1到最大的n位数

- 题目描述：输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。（用返回一个整数列表来代替打印，n为正整数）

- 示例：

```bash
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

- 代码1： `return list(range(1, 10 ** n))`，时间复杂度：`O（10**n）`， 空间复杂度：`O（1）`

- 代码2：由于int表示不了大数，我们考虑dfs 字串递归，1、11、111……

```py
def printnumsbers(n: int) -> List[int]:
    def dfs(index: int, nums: list, digit: int):
        # index: 当前index
        # nums: 缓存的数组
        # digit：总位数
        if index == digit:
            res.append(int(''.join(nums)))
            return
        for i in range(10):
            nums.append(str(i))
            dfs(index + 1, nums, digit)
            nums.pop()      # 回溯，弹出刚append的str(i)，恢复现场不影响下一次循环

    res = []
    for digit in range(1, n + 1):    # 这个循环由digit = 1开始，保证了有序
        for first in range(1, 10):
            nums = [str(first)]      # 将首位直接加入，可以避免dfs到001这种情况
            dfs(1, nums, digit)
    
    return res
```

### 丑数

- 题目描述：丑数 就是只包含质因数 2、3 和 5 的正整数。给你一个整数 n ，请你判断 n 是否为 丑数 。1被视为丑数，0不是。

- 示例：

```bash
输入：n = 6
输出：true
解释：6 = 2 × 3
```

- 代码：遍历factor除尽，时间复杂度：`O（logN）`（为什么是logN？每次至少除2），空间复杂度：`O（1）`

```py
def isUgly(n: int) -> bool:
    if n <= 0:
        return False

    # 为什么不需要一个大循环，除2、再除3再回过来除2？
    # 质因数的特性使可以先除2除尽再考虑3、5
    factors = [2, 3, 5]
    for factor in factors:
        while n % factor == 0:
            n //= factor
    
    return n == 1
```

## 中等

### 丑数二

- 题目描述：我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

- 示例：

```bash
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

- 代码：动态规划：时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def nthUglyNumber(n: int) -> int:
    # dp[i]表示第i个丑数，dp[1] = 1
    # 任意一个丑数，一定是由之前的丑数 * 2 或 * 3 或 * 5来的
    # p2表示*2指针，p3表示*3指针，p5表示*5指针。每使用一次，指针索引加一。
    # 例如p2=1时，dp[p2]*2 = 2，那么对于p2指向之处，*2就用掉了，之后就上p2=2，看下一个*2机会
    # dp[i] = min(dp[p2] * 2, dp[p3] * 3, dp[p5] * 5)
    dp = [0] * (n + 1)
    dp[1] = 1
    p2 = p3 = p5 = 1

    for i in range(2, n + 1):
        num2, num3, num5 = dp[p2] * 2, dp[p3] * 3, dp[p5] * 5
        dp[i] = min(num2, num3, num5)
        if dp[i] == num2:
            p2 += 1
        if dp[i] == num3:
            p3 += 1
        if dp[i] == num5:
            p5 += 1
    
    return dp[n]
```

### 礼物的最大价值

- 题目描述：在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

- 示例：

```bash
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

- 代码：动态规划。时间复杂度：`O（M*N）`，空间复杂度：`O（M*N）`

```py
def maxValue(grid: List[List[int]]) -> int:
    if len(grid) == 0 or len(grid[0]) == 0:
        return 0

    # dp[i][j] 表示到i、j格的最大价值
    # dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    # dp[0][0] = grid[0][0] 首行首列初始化依次累加
    h, w = len(grid), len(grid[0])
    dp = [[0] * (w) for _ in range(h)]
    dp[0][0] = grid[0][0]
    for i in range(1, h):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, w):
        dp[0][j] = dp[0][j-1] + grid[0][j]


    for i in range(1, h):
        for j in range(1, w):
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[-1][-1]

```

注：可以在grid上直接操作，使得空间复杂度降到O（1）

### 把数字翻译成字符串

- 题目描述：给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

- 示例：

```bash
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

- 代码：动态规划。时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def translateNum(num: int) -> int:
    if num < 10:
        return 1

    # dp长度为len(s), dp[i]为到s的i位置时，数字的翻译方法。
    # dp[0] = 1, dp[1] = 1 或 2
    # dp[i] = dp[i-1] if 新值不能与之前的值组成一个翻译方法
    # dp[i] = dp[i-1] + dp[i-2] if 新值可以与之前一位的值组成一个翻译方法
    # 解释：如果新值与前面一个值组成方法，那么方法数为dp[i-2]，如果不组成方法，方法数为dp[i-1]，求和
    # 最终返回dp[-1]
    s = str(num)
    dp = [0] * len(s)
    dp[0] = 1
    dp[1] = 2 if '10' <= s[0:2] <= '25' else 1

    for i in range(2, len(s)):
        dp[i] = dp[i-1] + dp[i-2] if '10' <= s[i-1:i+1] <= '25' else dp[i-1]

    return dp[-1]
```

### 把数组排成最小的数

- 题目描述：输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

- 示例：

```bash
输入: [5,2,3,10]
输出: "10235"
```

- 代码：更改排序规则进行排序——自定义快排。时间复杂度：`O（NlogN）`，空间复杂度：`O（N）`

```py
def minNumber(nums: List[int]) -> str:

    # 思路：将nums用一种新的方式排序，最后拼接
    # 新的方式是：如果 a + b > b + a， 那么b的值更小，放前面。
    # 例如3 + 30（330） > 30 + 3（303），30放前面
    def quick_sort(nums, low , high):
        if low >= high:
            return
        left, right = low, high
        key = nums[left]                  # 需要比较的基准值
        while left < right:
            # 右游标左移，直到找到小于key的值
            while left < right and key + nums[right] <= nums[right] + key:
                right -= 1
            nums[left] = nums[right]
            # 左游标右移，直到找到大于key的值
            while left < right and key + nums[left] >= nums[left] + key:
                left += 1
            nums[right] = nums[left]
        # 出循环，left = right
        nums[left] = key

        quick_sort(nums, low, left - 1)
        quick_sort(nums, left + 1, high)
    
    str_nums = [str(num) for num in nums]
    quick_sort(str_nums, 0, len(str_nums) - 1)
    return ''.join(str_nums)

```

- 注：此题要更严谨一些的话，可以证明一下传递性，a < b，b < c， 那么a < c
- 注2：我们可以用self存nums而不必压栈，减缓空间复杂度。

### 数组中数字出现的次数

- 题目描述：一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是`O(N)`，空间复杂度是`O(1)`。

- 示例：

```bash
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
```

- 思路：先考虑找出数组中唯一一个只出现一次的数字，其他都出现两次，异或直接解决。`[4, 1, 4] 即 4 ^ 1 ^ 4 = 1`。但我们这里有两个只出现一次的数字，所以要考虑**分组异或**，即让两个出现一次的数字分到不同组里，其他的出现两次的数字出现在同一组里。怎么分组呢？**先找到两个只出现一次的数字的不同点，即两个数字异或后的第一个二进制位不同的地方，然后以此分组**

- 代码：分组异或。时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def singleNumbers(nums: List[int]) -> List[int]:
    x, y = 0, 0
    n, m = 0, 1                 # n 存储 x^y， m 存储 x^y 的首位为1的值

    for num in nums:            # 1. 遍历异或，得到 n = x^y
        n ^= num

    while n & m == 0:           # 2. 循环左移，得到 x^y 首位为1 的值，如0010
        m <<= 1

    for num in nums:            # 分组，让x、y不在同一组里
        if num & m:             # a组，num & m == 1
            x ^= num
        else:                   # b组，num & m == 0
            y ^= num
    return x, y
```

### 数组中数字出现的次数 II

- 题目描述：在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

- 示例：

```bash
输入：nums = [3,4,3,3]
输出：4
```

- 代码1：哈希表遍历。时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def singleNumber(nums: List[int]) -> int:
    dic = defaultdict(int)
    for num in nums:
        dic[num] += 1
    
    for key, value in dic.items():
        if value == 1:
            return key
```

- 代码2：核心思路——对于出现三次的数字，各二进制出现的次数都是3的倍数，统计所有数字二进制并对3求余结果即为只出现一次的数字。时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def trainingPlan(actions: List[int]) -> int:
    counts = [0] * 32
    # 遍历统计第i位1的个数之和
    for action in actions:
        for i in range(32):
            counts[i] += action & 1
            action >>= 1

    # 取余然后每一位拼起来
    res, m = 0, 3
    for i in range(31, -1, -1):
        res <<= 1
        res |= counts[i] % m
    
    return res
```

### 二叉树中和为某一值的路径

- 题目描述：给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出**所有** 从根节点到叶子节点 路径总和等于给定目标和的路径。

- 示例：

```bash
输入：root = [1,2,3], targetSum = 5
输出：[]
输入：root = [1,2,3], targetSum = 4
输出：[1,3]
```

- 代码：**dfs**，时间复杂度：`O（N）`，空间复杂度：`O（K）`（result不算空间复杂度，path算，所以树高度为占用空间）

```py
def pathSum(root: TreeNode, target: int) -> List[List[int]]:
    result, path = [], []
    def dfs(node: TreeNode, target: int):
        if not node:
            return
        path.append(node.val)
        target -= node.val
        if target == 0 and not node.right and not node.left:  # 叶子节点且和为目标
            result.append(path.copy())      # 注意需要copy，因为path指向为同一片内存空间
        dfs(node.left, target)
        dfs(node.right, target)
        path.pop()                          # 标准回溯

    dfs(root, target)
    return result
```

### 二叉搜索树的后序遍历序列

- 题目描述：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

- 示例：

```bash
     5
    / \
   2   6
  / \
 1   3

输入: [1,6,3,2,5]
输出: false

     8
    / \
   5   9
  / \
 4   6

输入: [4,6,5,9,8]
输出: true
```

- 代码：递归分治：利用二叉搜索树后序遍历定义，最后一个是根，比根小的是左子树，比根大的是右子树。然后递归判断每一个子树。时间复杂度：`O（N^2）`（因为每次都是取一个根节点递归，如果二叉树退化成链表则恰为`O（N）`），空间复杂度：`O（logN）`（退化为链表为`O（N）`）

```py
def verifyPostorder(postorder: List[int]) -> bool:
    # 二叉搜索树定义：
    # 左子树中所有节点的值 < 根节点的值；右子树中所有节点的值 > 根节点的值；左右子树也为二叉搜索树
    def recur(low, high):
        if low >= high:      # 单结点，无需判断，直接true 
            return True
        # high为根节点值
        pointer = low 
        # 找到第一个大于根节点的值，因为没有重复值不需要考虑等于的情况
        while postorder[pointer] < postorder[high]:       
            pointer += 1
        # 此时poiner往左是左子树，往右是右子树
        mid = pointer

        # 让pointer右移
        while postorder[pointer] > postorder[high]:
            pointer += 1
        # 如果pointer走不到high，意味着不满足二叉搜索树的条件。如果走到递归判断
        return pointer == high and recur(low, mid - 1) and recur(mid, high - 1)

    return recur(0, len(postorder) - 1)
```

- 代码2：尝试拿列表直接构建BST（不需要真的构建），如果列表里值用完了说明符合。时间复杂度：`O（N）`，空间复杂度：`O（logN）`（栈递归）

```py
def verifyTreeOrder(self, postorder: List[int]) -> bool:
    # 构建方式：从列表最后一个（根）开始，依次递归右左子树
    def build(left: int, right: int):
        # 列表为空直接返回
        if not postorder:
            return
        val = postorder[-1]
        # 当前值不满足范围要求直接返回
        if not left < val < right:
            return
        # 移除列表的最后一个值(根节点)
        postorder.pop()
        # 递归构建右子树 右子树的上界不变 下界是当前节点
        build(val, right)
        # 递归构建左子树 左子树的下界不变 上界是当前节点
        build(left, val)

    build(-sys.maxsize, sys.maxsize)
    # 构造结束列表不为空 说明不是合法的后序遍历序列
    return not postorder
```

### 最长不含重复字符的子字符串

- 题目描述：请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。注意是子串，子串是相连的。

- 示例：

```bash
输入: "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

- 代码：双指针滑动窗口。时间复杂度：`O（N）`，空间复杂度：`O（1）`

```py
def lengthOfLongestSubstring(s: str) -> int:

    left, right = 0, 0
    dic = defaultdict(int)
    l = len(s)
    max_l, count = 0, 0

    while right < l:
        # 遇到重复字符，左指针右移，直到排除这个字符
        while dic[s[right]] == 1:
            dic[s[left]] -= 1
            left += 1
            count -= 1
        # 没有重复字符，右指针右移，扩大范围
        dic[s[right]] += 1
        count += 1
        right += 1
        if count > max_l:
            max_l = count
    
    return max_l
```

### 第 N 位数字

- 题目描述：给你一个整数 `n` ，请你在无限的整数序列 `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]` 中找出并返回第 `n` 位上的数字。

- 示例：

```bash
输入：n = 11
输出：0
解释：第 11 位数字在序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是 0 ，它是 10 的一部分。
```

- 思路：直接计算，找规律，统计各个阶段数位、数字、数量。时间复杂度：`O（log10 N）`，空间复杂度：`O（1）`

```py
def findNthDigit(self, n: int) -> int:
    # 分别表示当前阶段（每个数字有多少位），该阶段起始点，该阶段数位量
    digit, start, count = 1, 1, 9

    while n > count: # 找到对应阶段，如16，位于第二阶段
        n -= count
        start *= 10     # 第二阶段开始值为10
        digit += 1      # 第二阶段数位为2
        count = 9 * start * digit       # 第二阶段总数量为9 * start * digit（180）
    # 找到对应数字，如16，对应数字为10+ (16-9) // 2 =13
    num = start + (n - 1) // digit
    return int(str(num)[(n - 1) % digit])       # 13对应数位为为第一位
```

### 从上到下打印二叉树

- 题目描述：从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

- 代码：层序遍历，用一个队列辅助。时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def levelOrder(root: TreeNode) -> List[int]:
    queue = deque()
    queue.append(root)
    result = []

    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
    
    return result
```

### 从上到下打印二叉树 II

- 题目描述：从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

- 示例：

```c
[
  [3],
  [9,20],
  [15,7]
]
```

- 代码：层序遍历+分层（一次清空队列）。时间空间复杂度都为`O（N）`

```py
def levelOrder(root: TreeNode) -> List[List[int]]:
    queue = deque()
    queue.append(root)
    result = []

    while queue:
        tmp = []
        for _ in range(len(queue)):
            node = queue.popleft()
            if node:
                tmp.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        if len(tmp) > 0:
            result.append(tmp)
        
    return result
```

### 从上到下打印二叉树 III

- 题目描述：请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

- 示例（上题同样，输出不同）

```c
[
  [3],
  [20,9],
  [15,7]
]
```

- 代码：层序遍历 + 分层 + 部分反序。时间复杂度：`O（N）`，空间复杂度：`O（N）`

```py
def levelOrder(root: TreeNode) -> List[List[int]]:
    queue = deque()
    queue.append(root)

    result = []
    level = 0
    while queue:
        one_layer = []
        for _ in range(len(queue)):
            node = queue.popleft()
            if node:
                one_layer.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        
        if len(one_layer) > 0:
            result.append(one_layer[::-1] if level & 1 else one_layer)
        level += 1
    
    return result
```






### 字符串的排列

- 题目描述：输入一个字符串，打印出该字符串中字符的所有排列。
- 示例：
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]

- 代码：

```py
def permutation(self, s: str) -> List[str]:
    # 统计字符数
    dic = defaultdict(int)
    for c in s:
        dic[c] += 1
    
    result = []
    l = len(s)
    # dfs遍历所有排列组合，此时不需要剪枝因为每一次迭代都是有效组合
    def dfs(s, level):
        for next_key in dic.keys():
            tmp = s     # 这里需要使用tmp而不是s，避免s被其他干扰
            if dic[next_key] > 0:
                tmp += next_key
                dic[next_key] -= 1
                if level == l:
                    result.append(tmp)
                dfs(tmp, level+1)
                dic[next_key] += 1          # 回溯

    dfs('', 1)

    return result
```

### 栈的压入、弹出序列

- 题目描述：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

- 代码：

```py
def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
    # 思路：辅助栈贪心，每次取要么是新放入栈的取，要么是刚好栈顶就是
    push_index = 0
    stack = []
    for num in popped:
        # 栈顶不是对应元素，新放入栈
        while len(stack) == 0 or stack[-1] != num:
            # 所有元素入栈，但还没能走完poped，说明无法匹配
            if push_index == len(pushed):
                return False
            stack.append(pushed[push_index])
            push_index += 1
        
        # 找到栈顶是该元素
        stack.pop()
    
    return True
```

- 时间复杂度：O（N），空间复杂度：O（N）

### 二叉搜索树与双向链表

- 题目描述：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

- 示例：见[leetcode](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

- 代码：前序遍历，准备一个pre指针指向前一个访问过的结点。

```py
class Solution:

    def __init__(self):
        # 使用self可以直接存变量，比递归中用参数存变量方便
        self.pre = self.head = None

    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return root

        def recur(cur) -> None:
            if not cur:
                return

            recur(cur.left)
            if self.pre:        # 正常情况
                cur.left, self.pre.right = self.pre, cur
            else:               # 首次情况
                self.head = cur
            self.pre = cur
            recur(cur.right)

        recur(root)

        # 出循环后，pre恰好指到尾结点
        self.pre.right, self.head.left = self.head, self.pre    # 最终头尾相连

        return self.head
```

### 复杂链表的复制

- 题目说明：请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

- 示例：

```c
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

注：显然不能指向同一片内存空间，而应该是新创建的

- 切入：难点在于random指向任意结点，那么显然双指针就行不通了。
- 代码1——**哈希表，“原节点 -> 新节点” 的映射**：

时间复杂度：O（N），空间复杂度：O（N）

```py
def copyRandomList(self, head: 'Node') -> 'Node':
    if not head:
        return
    dic = {}

    # 1. 复制创建新节点，并建立 “原节点 -> 新节点” 的 Map 映射
    cur = head
    while cur:
        dic[cur] = Node(cur.val)
        cur = cur.next

    cur = head
    # 2. 构建新节点的 next 和 random 指向
    while cur:
        dic[cur].next = dic.get(cur.next)
        dic[cur].random = dic.get(cur.random)
        cur = cur.next

    # 3. 返回新链表的头节点
    return dic[head]

```

- 代码2——**构建新链表，原节点 1 -> 新节点 1 -> 原节点 2 -> 新节点 2 -> … ，加上random后再拆分**：

时间复杂度：O（N），空间复杂度：O（1）（构建答案不算空间复杂度）

```py
def copyRandomList(self, head: 'Node') -> 'Node':
    if not head:
        return
    
    # 1. 复制各节点，并构建拼接链表
    cur = head
    while cur:
        tmp = Node(cur.val)
        tmp.next = cur.next
        cur.next = tmp
        cur = tmp.next

    # 2. 构建各新节点的 random 指向
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next

    # 3. 拆分两链表
    cur = res = head.next
    pre = head
    while cur.next:
        pre.next = pre.next.next
        cur.next = cur.next.next
        pre = pre.next
        cur = cur.next
    pre.next = None # 单独处理原链表尾节点(这样对原链表没有影响。你的函数不应该影响原链表)
    return res      # 返回新链表头节点
```

### 二维数组的查找

- 题目说明：在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

0 <= n <= 1000
0 <= m <= 1000

- 示例：

现有矩阵 matrix 如下：

```c
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

给定 target = 5，返回 true。
给定 target = 20，返回 false。

- 代码——**往左往下走**：

```py
def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return False
    h, w = len(matrix), len(matrix[0])
    x, y = w-1, 0       # 从右上角开始，保证不会漏
    while x >= 0 and y < h:
        if matrix[y][x] == target:
            return True
        elif matrix[y][x] > target:
            x = x-1
        else:
            y = y + 1
    return False
```

- 时间复杂度：O（N+M），空间复杂度：O（1）
- 注：如果从左上角开始就有可能漏数，比如点位在下方却往右走了。而左下或右上开始，可以很容易证明路径唯一。
- 证明：如果当前元素大于目标值，说明当前元素的下边的所有元素都一定大于目标值，因此往下查找不可能找到目标值，往左查找可能找到目标值（左下也在左）。如果当前元素小于目标值，说明当前元素的左边的所有元素都一定小于目标值，因此往左查找不可能找到目标值，往下查找可能找到目标值。

### 矩阵中的路径（再）

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

- 示例：

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

- 代码：

```py
def exist(self, board: List[List[str]], word: str) -> bool:
        
    w, h = len(board[0]), len(board)
    l = len(word)
    def dfs(x, y, k):
        if not 0 <= x < w or not 0 <= y < h or board[y][x] != word[k]:
            return False
        if k == l - 1:
            return True
        board[y][x] = ''    # 标记在原数组就不需要标记数组
        for nx, ny in [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]:
            if dfs(nx, ny, k+1):
                return True
        board[y][x] = word[k]   # 很重要！将结果回溯

    for x in range(w):
        for y in range(h):
            if dfs(x, y, 0):
                return True
    return False
```

- 时间复杂度：O（MN * 3^k)，MN为开启dfs次数，3^k为最坏情况下的搜索次数（每个都搜索3次，搜索k层）
- 空间复杂度：O（k）或者O（1），k为栈最多调用次数

### 机器人的运动范围（再）

- 题目说明：地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

- 示例：
输入：m = 2, n = 3, k = 1
输出：3

- 代码：dfs

```py
def movingCount(self, m: int, n: int, k: int) -> int:
    
    def digit_sum(x: int):
        # 通用计算数位和方法
        result = 0
        while x != 0:
            result += x % 10
            x = x // 10
        return result

    marks = [[0] * n for _ in range(m)]

    def dfs(x: int, y: int):
        # 实际证明，dfs先判断更好
        if not (0 <= x < n and 0 <= y < m and (digit_sum(x)+digit_sum(y))<=k and marks[y][x] != 1):
            return 0
        marks[y][x] = 1
        return 1 + dfs(x+1, y) + dfs(x, y+1) + dfs(x-1, y) + dfs(x, y-1)
    
    return dfs(0, 0)
```

- 时间复杂度：O（MN），空间复杂度：O（MN）

### 重建二叉树（再）

- 题目描述：输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。假设不含重复数字

- 示例：
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

- 代码：

```py
def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    # 核心思路：递归，找到中序（先左子树）的根，划分中序结果为左、根、右，再把前序划分为根、左、右即可

    def recur(pre_root: int, in_left: int, in_right: int):
        if in_left > in_right:    # 递归终止
            return                               
        node = TreeNode(preorder[pre_root])
        in_root = dic[preorder[pre_root]]
        node.left = recur(pre_root + 1, in_left, in_root - 1)
        # in_root - in_left + pre_root + 1 解释：找到preorder中右子树根的位置
        node.right = recur(in_root - in_left + pre_root + 1, in_root + 1, in_right)
        return node

    dic = {}
    for i in range(len(inorder)):   # 哈希表以快速获取index，只有在无重复值时能用
        dic[inorder[i]] = i
    return recur(0, 0, len(inorder) - 1)

```

- 时间复杂度：O（N），空间复杂度：O（N）

### 剪绳子-1（再）

- 题目描述：
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

- 示例：
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1

- 代码：

```py
def cuttingRope(self, n: int) -> int:
    # 没有思路的话，考虑先推几个例子，比如1、2、3 ... 12你就发现越多化成3的片段，值越大，那就很简单了
    # 本题用动态规划做：想知道n的最大划分，从<n的绳子转移而来

    '''
    用dp[i]表示长度为i的绳子剪成m段后的最大乘积，初始化dp[2] = 1
    我们先把绳子剪掉第一段（长度为j），如果只剪掉长度为1，对最后的乘积无任何增益，所以从长度为2开始剪
    剪了第一段后，剩下(i - j)长度可以剪也可以不剪。如果不剪的话长度乘积即为j * (i - j)；
    如果剪的话长度乘积即为j * dp[i - j]。取两者最大值max(j * (i - j), j * dp[i - j])
    第一段长度j可以取的区间为[2,i)，对所有j不同的情况取最大值，因此最终dp[i]的转移方程为
    dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
    '''

    dp = [0] * (n + 1)
    dp[2] = 1
    for i in range(3, n + 1):
        for j in range(2, i):
            dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
    return dp[n]
```

- 时间复杂度：O（N^2），空间复杂度：O（N）

- 代码2：看穿了尽可能划分成3后的代码，时空复杂度均为O（1）

```py
def cuttingRope(self, n: int) -> int:
    if n < 4:
        return n-1
    result = 1
    while n>4:
        result *= 3
        n -= 3
    return result*n
```

### 树的子结构(再)

- 题目描述：输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
B是A的子结构， 即 A中有出现和B相同的结构和节点值。

- 示例：
输入：A = [1,2,3], B = [3,1]
输出：false
输入：A = [3,4,5,1,2], B = [4,1]
输出：true

- 思路：先根遍历找到B的根，然后A、B开始一起遍历，如果A为空或者A、B值不相同都不符合，如果B为空则符合。

- 代码：

```py
def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:

    if A is None or B is None:
        return False

    def recur(A, B):
        if not B:
            return True
        if not A or A.val != B.val:
            return False
        return recur(A.left, B.left) and recur(A.right, B.right)

    if recur(A, B):
        return True
    
    return self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)
```

- 时间复杂度：O（MN）（注：MN为A、B结点数量，先遍历A占用O（M），recur占用O（N）
- 空间复杂度：O（M）（最大递归深度）

### 表示数值的字符串（再）

- 题目描述：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。

数值（按顺序）可以分成以下几个部分：

1. 若干空格
2. 一个 小数 或者 整数
3. （可选）一个 'e' 或 'E' ，后面跟着一个 整数
4. 若干空格

小数（按顺序）可以分成以下几个部分：

1. （可选）一个符号字符（'+' 或 '-'）
2. 下述格式之一：
    至少一位数字，后面跟着一个点 '.'
    至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
    一个点 '.' ，后面跟着至少一位数字

整数（按顺序）可以分成以下几个部分：

1. （可选）一个符号字符（'+' 或 '-'）
2. 至少一位数字

部分数值列举如下：
["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
部分非数值列举如下：
["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]

- 示例：
输入：s = "0"
输出：true
输入：s = "    .1  "
输出：true

- 思路：输入一个字符串，判断是不是xx的题目，我们可以统一用编译原理中的**有限状态自动机**来做。python中直接`try float() expect`是可以的，直接利用了python解释器中的有限状态自动机。

- 代码：

官方代码，使用enum便于开发，时间复杂度:O(N)，空间复杂度：O（1），因为enum所以速度较慢
![图片](resources/有限状态自动机1.png)

```py
from enum import Enum

class Solution:
    def isNumber(self, s: str) -> bool:
        State = Enum("State", [
            "STATE_INITIAL",
            "STATE_INT_SIGN",
            "STATE_INTEGER",
            "STATE_POINT",
            "STATE_POINT_WITHOUT_INT",
            "STATE_FRACTION",
            "STATE_EXP",
            "STATE_EXP_SIGN",
            "STATE_EXP_NUMBER",
            "STATE_END"
        ])
        Chartype = Enum("Chartype", [
            "CHAR_NUMBER",
            "CHAR_EXP",
            "CHAR_POINT",
            "CHAR_SIGN",
            "CHAR_SPACE",
            "CHAR_ILLEGAL"
        ])

        def toChartype(ch: str) -> Chartype:
            if ch.isdigit():
                return Chartype.CHAR_NUMBER
            elif ch.lower() == "e":
                return Chartype.CHAR_EXP
            elif ch == ".":
                return Chartype.CHAR_POINT
            elif ch == "+" or ch == "-":
                return Chartype.CHAR_SIGN
            elif ch == " ":
                return Chartype.CHAR_SPACE
            else:
                return Chartype.CHAR_ILLEGAL
        
        transfer = {
            State.STATE_INITIAL: {
                Chartype.CHAR_SPACE: State.STATE_INITIAL,
                Chartype.CHAR_NUMBER: State.STATE_INTEGER,
                Chartype.CHAR_POINT: State.STATE_POINT_WITHOUT_INT,
                Chartype.CHAR_SIGN: State.STATE_INT_SIGN
            },
            State.STATE_INT_SIGN: {
                Chartype.CHAR_NUMBER: State.STATE_INTEGER,
                Chartype.CHAR_POINT: State.STATE_POINT_WITHOUT_INT
            },
            State.STATE_INTEGER: {
                Chartype.CHAR_NUMBER: State.STATE_INTEGER,
                Chartype.CHAR_EXP: State.STATE_EXP,
                Chartype.CHAR_POINT: State.STATE_POINT,
                Chartype.CHAR_SPACE: State.STATE_END
            },
            State.STATE_POINT: {
                Chartype.CHAR_NUMBER: State.STATE_FRACTION,
                Chartype.CHAR_EXP: State.STATE_EXP,
                Chartype.CHAR_SPACE: State.STATE_END
            },
            State.STATE_POINT_WITHOUT_INT: {
                Chartype.CHAR_NUMBER: State.STATE_FRACTION
            },
            State.STATE_FRACTION: {
                Chartype.CHAR_NUMBER: State.STATE_FRACTION,
                Chartype.CHAR_EXP: State.STATE_EXP,
                Chartype.CHAR_SPACE: State.STATE_END
            },
            State.STATE_EXP: {
                Chartype.CHAR_NUMBER: State.STATE_EXP_NUMBER,
                Chartype.CHAR_SIGN: State.STATE_EXP_SIGN
            },
            State.STATE_EXP_SIGN: {
                Chartype.CHAR_NUMBER: State.STATE_EXP_NUMBER
            },
            State.STATE_EXP_NUMBER: {
                Chartype.CHAR_NUMBER: State.STATE_EXP_NUMBER,
                Chartype.CHAR_SPACE: State.STATE_END
            },
            State.STATE_END: {
                Chartype.CHAR_SPACE: State.STATE_END
            },
        }

        st = State.STATE_INITIAL
        for ch in s:
            typ = toChartype(ch)
            if typ not in transfer[st]:
                return False
            st = transfer[st][typ]
        
        return st in [State.STATE_INTEGER, State.STATE_POINT, State.STATE_FRACTION, State.STATE_EXP_NUMBER, State.STATE_END]
```

某大佬代码，直接使用dic，较快

![图片](resources/有限状态自动机2.png)

```py
class Solution:
    def isNumber(self, s: str) -> bool:
        states = [
            { ' ': 0, 's': 1, 'd': 2, '.': 4 }, # 0. start with 'blank'
            { 'd': 2, '.': 4 } ,                # 1. 'sign' before 'e'
            { 'd': 2, '.': 3, 'e': 5, ' ': 8 }, # 2. 'digit' before 'dot'
            { 'd': 3, 'e': 5, ' ': 8 },         # 3. 'digit' after 'dot'
            { 'd': 3 },                         # 4. 'digit' after 'dot' (‘blank’ before 'dot')
            { 's': 6, 'd': 7 },                 # 5. 'e'
            { 'd': 7 },                         # 6. 'sign' after 'e'
            { 'd': 7, ' ': 8 },                 # 7. 'digit' after 'e'
            { ' ': 8 }                          # 8. end with 'blank'
        ]
        p = 0                           # start with state 0
        for c in s:
            if '0' <= c <= '9': t = 'd' # digit
            elif c in "+-": t = 's'     # sign
            elif c in "eE": t = 'e'     # e or E
            elif c in ". ": t = c       # dot, blank
            else: t = '?'               # unknown
            if t not in states[p]: return False
            p = states[p][t]
        return p in (2, 3, 7, 8)
```

### 数值的整数次方（再）

- 题目说明：实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

- 示例：
输入：x = 2.00000, n = -2
输出：0.25000
输入：x = 0, n = 0
输出：1

- 代码：（快速幂法）

```py
def myPow(self, x: float, n: int) -> float:
    # 注意n为整形，不会出现浮点
    if n == 0:
        return 1
    if x == 0:
        return 0
    res = 1
    if n < 0:
        x, n = 1 / x, -n

    # x为奇数时，x ^ n = x * (x^2)^(n/2)
    # x为偶数时，x ^ n = (x^2) ^ (n/2)
    # 最后不断展开，n化为0的时候退出循环。
    while n:
        if n & 1:   # n为奇数的时候
            res *= x
        x *= x
        n >>= 1
    return res
```

- 时间复杂度：O（logn），空间复杂度：O（1）

## 困难

### 滑动窗口的最大值

- 题目描述：给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

- 示例：
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7]
解释:

滑动窗口的位置                最大值
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

- 代码——如果暴力法时间复杂度为O（NK），我们使用队列存储数值，让获取最大值的行为变为O（1）。备注：在滑动窗口踢出现有最大值的时候，能立刻获得下一个最大值。
时间复杂度：O（N），空间复杂度：O（N）

```py
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or k == 0:
            return []

        # deque队首存放当前窗口最大值
        deque = collections.deque()
        # 未形成窗口前
        for i in range(k):
            # 新元素进来，如果比之前元素都大，那么清零之前元素，并保证队首存放最大值
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
        res = [deque[0]]

        # 形成窗口后
        for i in range(k, len(nums)):
            # 如果最大元素被踢出窗口，popleft
            if deque[0] == nums[i - k]:
                deque.popleft()
            # 保证最大元素在队首
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
            res.append(deque[0])
        return res
```

### 不用加减乘除做加法（可选，再）

- 题目描述：写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

- 代码：二进制运算
时间复杂度：O（1），空间复杂度：O（1）

```py
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a & x, b & x     # 将数字a、b截断，只保留三十二位
        while b != 0:
            a, b = a ^ b, ((a & b) << 1) & x
            '''addition = ((a & b) << 1) & x      # a和b产生的进位，注意截断高位
            a = a ^ b                  # 二进制加，如果不考虑进位，那么就相当于异或，把结果赋给a
            b = addition               # 进位赋给b'''
            # 接下来要处理的就是，上一轮运算的结果，和产生的进位再进行“加”运算
        
        # 进位为0时出循环
        # 如果a为正数，那么直接返回a
        # 如果a为负数，先和x异或让三十二位按位取反，再对整个数字取反（包括三十二位之前的）转成python存储负数的格式。
        # 注意：对于python而言，负数三十二位之前，无限补1
        return a if a <= 0x7fffffff else ~(a ^ x)
```

### 数组中的逆序对（再）

- 题目描述：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
- 示例：
输入: [7,5,6,4]
输出: 5

- 思路：归并排序，归并的时候进行统计，很自然就得到了逆序对数
- 代码：
时间复杂度：O（nlogn），空间复杂度：O（N）（注：递归分治，同时只有一侧函数在执行，栈最多占用n/2 + n/4 + n/8 ……这部分空间）

```py
class Solution:
    def __init__(self):
        self.count = 0

    def reversePairs(self, nums: List[int]) -> int:
        def merge(lst_left: list, lst_right: list) -> list:
            left, right = 0, 0
            left_l, right_l = len(lst_left), len(lst_right)

            # 逆序merge，这样大的值在前面，方便统计逆序对
            new_list = []
            while left < left_l and right < right_l:
                if lst_left[left] > lst_right[right]:
                    new_list.append(lst_left[left])
                    left += 1
                    self.count += right_l - right       # 因为逆序，所以直接加上之后所有的逆序对
                else:
                    new_list.append(lst_right[right])
                    right += 1
            # 剩余列表直接extend至列表尾
            new_list += lst_left[left:]
            new_list += lst_right[right:]
            return new_list


        def merge_sort(lst: list) -> list:
            if len(lst) <= 1:
                return lst

            mid = len(lst) // 2
            lst_left = merge_sort(lst[:mid])
            lst_right = merge_sort(lst[mid:])
            return merge(lst_left, lst_right)

        merge_sort(nums)

        return self.count
```

### 1～n 整数中 1 出现的次数（可选，再）

- 题目描述：输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

- 代码：计算每一位上1出现次数的和

时间复杂度：O（log10 N），空间复杂度：O（1）

```py
def countDigitOne(self, n: int) -> int:
    # 核心思路：计算每一位上1出现次数的和

    digit, res = 1, 0                       # 当前数位，结果
    high, cur, low = n // 10, n % 10, 0     # 高位数，当前数，当前数后数。eg 2302，比如cur指向0，高位23
    while high != 0 or cur != 0:
        if cur == 0:
            # 当前位为0，公式：digit * high，如209，十位上出现2*10个1 = 20个1
            res += high * digit             
        elif cur == 1:
            # 当前位为1，公式：digit * high + low + 1，如216，十位上出现2*10 + 6 + 1个1 = 27个1
            res += high * digit + low + 1   
        else:
            # 当前位大于1，公式：digit * (high + 1)，如220，十位上出现(2+1)*10 = 30个1
            res += (high + 1) * digit
        low += cur * digit
        cur = high % 10
        high //= 10
        digit *= 10
    return res
```

### 序列化二叉树

- 题目描述：请实现两个函数，分别用来序列化和反序列化二叉树。
- 示例：
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]
- 代码：
时间复杂度：O（N），空间复杂度：O（N）

```py
class Codec:

    def serialize(self, root):
        # 层次遍历
        queue = deque()
        queue.append(root)
        result = []
        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append('null')
        
        # 删掉末尾的None
        while len(result) > 0 and result[-1] is 'null':
            result.pop()

        return '[' + ','.join(result) + ']'

    def _consturct(self, node, index):
        left, right = 2 * index + 1, 2 * index + 2
        if left < len(self.data):
            node.left = TreeNode(self.data[left])
            self._consturct(node.left, left)
        if right < len(self.data):
            node.right = TreeNode(self.data[right])
            self._consturct(node.right, right)
        
    def deserialize(self, data):
        # 注意：这里同样可以用层序遍历，append root，弹出root加上左右子结点
        if data == '[]':
            return None
        data = data[1:-1].split(',')
        self.data = data
        root = TreeNode(data[0])
        self._consturct(root, 0)
        return root
```

### 正则表达式匹配（再，可选）

- 题目描述：给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
- 示例：
输入：s = "aab" p = "c*a*b"
输出：true
解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。

- 代码（递归）：利用lru_cache，本质是动态规划，但实现起来更简单。

```py
def isMatch(self, s: str, p: str) -> bool:

    len_s, len_p = len(s), len(p)

    # 装饰符实现记忆化搜索，等价于Top-Down动态规划
    @lru_cache(None)
    def recur(i: int, j: int) -> bool:
        # 结束条件
        if j == len_p:
            return i == len_s

        # 当前字母匹配
        first_match = (len_s > i) and (p[j] == s[i] or p[j] == '.')

        # 如果有`*`，那么两种情况
        # a、接受当前字母不匹配，相当于*取0
        # b、当前字母匹配，由于*可以取多个，那么直接走下一个recur  
        if len_p >= j+2 and p[j+1] == '*':
            return recur(i, j+2) or (first_match and recur(i+1, j))

        # 一般情况，当前字母和pattern index均+1
        return first_match and recur(i+1, j+1)

    return recur(0, 0)
```

- 时间复杂度：O（MN）（最坏情况），空间复杂度：O(MN)（最坏情况）

### 数据流中的中位数（再）

- 题目描述：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。

- 示例
输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]

- 代码1：单一列表，查中位数时排序后返回中位数。时间复杂度O（NlogN），空间复杂度O（N）
- 代码2：两个堆，一个大根堆，一个小根堆。时间复杂度：O（logN-维持堆的不变性），空间复杂度：O（N）

```py
from heapq import *

class MedianFinder:
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.B, -heappushpop(self.A, num))
        else:
            heappush(self.A, -heappushpop(self.B, -num))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0

```
