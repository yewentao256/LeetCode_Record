# 剑指offer2

## 简单

### 删除链表的节点

- 题目描述：给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回头结点
- 示例：

输入: head = [4,5,1,9], val = 5
输出: [4,1,9]

- 代码：

```py
def deleteNode(self, head: ListNode, val: int) -> ListNode:
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

- 时间复杂度：O（N），空间复杂度：O（1）

### 反转链表（再）

- 题目描述：定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

- 示例：
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL

- 代码：

```py
def reverseList(self, head: ListNode) -> ListNode:

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

- 时间复杂度：O（1），因为弹出操作对每个元素最多只操作一次，因此均摊下来为O（1）；空间复杂度：O（N）

### 斐波那契数列

- 题目说明：略

- 代码：

```py
def fib(self, n: int) -> int:
    MOD = 10 ** 9 + 7
    if n < 2:
        return n
    p, q, r = 0, 0, 1
    for i in range(2, n + 1):
        p = q
        q = r
        r = p + q
    return r % MOD
```

- 时间复杂度：O（N），空间复杂度：O（1）
- 注：如果用矩阵快速幂的方法可以将时间复杂度降到O（logn）
- 注：先算好再给的方法，初始化O（N）之后，每次fib都是O（1）的时间复杂度

### 数组中重复的数字

- 题目说明：找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

- 示例：

输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3

- 代码：

```py
def findRepeatNumber(self, nums: List[int]) -> int:
    s = set()
    l = 0
    for num in nums:
        s.add(num)
        l += 1
        if len(s) != l:
            return num
```

- 时间复杂度：O（N），空间复杂度：O（N）（会比dict稍微好一点）

### 青蛙跳台阶

- 题目说明：一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

- 示例：输入：n = 0；输出：1

- 代码：

```py
def numWays(self, n: int) -> int:
    MOD = 10 ** 9 + 7
    lst = [1, 1]
    for i in range(2, n+1):
        new_ele = (lst[i-1]+lst[i-2]) % MOD
        lst.append(new_ele)
    return lst[n]
```

- 时间复杂度：O（N），空间复杂度：O（N）
- 注：也可以使用循环变量法，`a, b = b, a + b`，使得空间复杂度降至O（1）

### 旋转数组的最小数字

- 题目说明：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

- 示例：
输入：[3,4,5,1,2]
输出：1

- 代码（二分查找）：

```py
def minArray(self, numbers: List[int]) -> int:

    left, right = 0, len(numbers) - 1
    while left < right:
        p = (left+right) // 2
        if numbers[p] > numbers[right]:     # 因为numbers[p]更大，不可能是最小的
            left = p + 1
        elif numbers[p] < numbers[right]:
            right = p
        else:
            right -= 1
    
    return numbers[left]
```

- 时间复杂度：O（N），空间复杂度：O（1）

### 替换空格

- 题目描述：请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

- 示例：略

- 代码：

```py
def replaceSpace(self, s: str) -> str:
        return '%20'.join(s.split(' '))
```

- 时间复杂度：O（N），空间复杂度：O（N）
- 注：可以通过其他方式比如扩充字串呀什么的实现O（1）空间复杂，但没有必要

### 从头到尾打印列表

- 题目说明：输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

- 示例：略

- 代码：

```py
def reversePrint(self, head: ListNode) -> List[int]:
    
    stack = []
    while head:
        stack.append(head.val)
        head = head.next
    return stack[::-1]
```

- 时间复杂度：O(N), 空间复杂度：O（N）

### 合并两个排序的链表

- 题目说明：输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

- 示例：
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

- 代码：

```py
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

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

- 时间复杂度：O（N）或者说O（N1+N2），空间复杂度：O（1）

### 二叉树的镜像

- 题目说明：请完成一个函数，输入一个二叉树，该函数输出它的镜像。

- 示例：
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]

- 代码：

```py
def mirrorTree(self, root: TreeNode) -> TreeNode:
    if not root:
        return
    root.left, root.right = root.right, root.left
    
    self.mirrorTree(root.left)
    self.mirrorTree(root.right)

    return root
```

- 时间复杂度：O（N），空间复杂度：O（N）（栈深。最坏情况下，二叉树退化为链表）

### 对称的二叉树（再）

- 题目说明：请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

- 示例：
输入：root = [1,2,2,3,4,4,3]
输出：true
输入：root = [1,2,2,null,3,null,3]
输出：false

- 代码：

```py
def isSymmetric(self, root: TreeNode) -> bool:
    # 同时放入左右子树，就可以轻松做对比了
    def recur(L, R):
        if not L and not R:
            return True
        if not L or not R or L.val != R.val:
            return False
        return recur(L.left, R.right) and recur(L.right, R.left)

    return recur(root.left, root.right) if root else True
```

- 时间复杂度：O（N），空间复杂度：O（N）（最差情况下，三角延伸）

### 调整数组顺序使奇数位于偶数前面

- 题目说明：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

- 示例：
输入：nums = [1,2,3,4]
输出：[1,3,2,4]
注：[3,1,2,4] 也是正确的答案之一。

- 代码

```py
def exchange(self, nums: List[int]) -> List[int]:
    left, right = 0, len(nums)-1

    while left < right:
        # 找到左指针指向为奇数的
        while left<right and nums[left] &1 == 1:    # 按位与& 比%要快挺多
            left += 1
        # 找到右指针指向为奇数的
        while left<right and nums[right] &1 ==0:
            right -= 1

        nums[left], nums[right] = nums[right], nums[left]
        left +=1
    return nums
```

- 时间复杂度：O（N），空间复杂度：O（1）

### 二进制中1的个数

- 题目说明：编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。

- 示例：
输入：n = 11 (控制台输入 00000000000000000000000000001011)
输出：3

- 代码：

```py
def hammingWeight(self, n: int) -> int:
    res = 0
    while n:
        res += n & 1
        n >>= 1
    return res
```

- 时间复杂度：O（$log_2{n}$），空间复杂度：O（1）

### 顺时针打印矩阵

- 题目描述：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

- 示例：
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]

- 代码：

```py
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []

    # 可以访问的边界，左、上、右、下
    left, up, right, down = 0, 0, len(matrix[0])-1, len(matrix)-1
    result = []
    while True:
        # 左到右
        for i in range(left, right+1):
            result.append(matrix[up][i])
        up += 1
        if up>down:
            break

        # 上到下
        for j in range(up, down+1):
            result.append(matrix[j][right])
        right -= 1
        if left>right:
            break

        # 右到左
        for i in range(right, left-1, -1):
            result.append(matrix[down][i])
        down -= 1
        if up>down:
            break


        # 下到上
        for j in range(down, up-1, -1):
            result.append(matrix[j][left])
        left += 1
        if left>right:
            break
    return result
```

- 时间复杂度：O（MN），空间复杂度：O（1）

### 链表中倒数第k个节点

- 题目描述：输入一个链表，输出该链表中倒数第k个节点。
- 示例：
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
- 代码：

```py
def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
    right = left = head
    
    for i in range(k):
        right = right.next
    
    while right:
        left = left.next
        right = right.next
    
    return left
```

- 时间复杂度：O（N），空间复杂度：O（1）

### 打印从1到最大的n位数

- 题目描述：输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。（用返回一个整数列表来代替打印，n为正整数）

- 示例：
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]

- 代码1： `return list(range(1, 10 ** n))`
- 时间复杂度：O（10**n）， 空间复杂度：O（1）

- 代码2：考虑到大数问题，int无法很好地保存数值，应该使用字符串保存（本题不要求）

```py
def printnumsbers(self, n: int) -> List[int]:
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
            nums.pop()      # 回溯，弹出刚append的str(i)

    res = []
    for digit in range(1, n + 1):
        for first in range(1, 10):
            nums = [str(first)]      # 将首位直接加入，可以避免dfs到001这种情况，还要去除。
            dfs(1, nums, digit)
    
    return res
```

## 中等

### 复杂链表的复制（再）

- 题目说明：请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

- 示例：
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
注：显然不能指向同一片内存空间，而应该是新创建的

- 切入：难点在于random指向任意结点，那么显然双指针就行不通了。
- 代码1（哈希表，“原节点 -> 新节点” 的映射）：

时间复杂度：O（N），空间复杂度：O（N）

```py
def copyRandomList(self, head: 'Node') -> 'Node':
    if not head:
        return
    dic = {}

    # 1. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
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

- 代码2（构建新链表，原节点 1 -> 新节点 1 -> 原节点 2 -> 新节点 2 -> … ，加上random后再拆分）：

时间复杂度：O（N），空间复杂度：O（1）（构建答案不算空间复杂）

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

### 二维数组的查找（再）

- 题目说明：在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

0 <= n <= 1000
0 <= m <= 1000

- 示例：

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。
给定 target = 20，返回 false。

- 代码：

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
- 注：如果从左上角开始就有可能漏数，比如点位在下方却往右走了。而左下或右上开始，可以很容易证明路径唯一。（如果当前元素大于目标值，说明当前元素的下边的所有元素都一定大于目标值，因此往下查找不可能找到目标值，往左查找可能找到目标值（左下也在左）。如果当前元素小于目标值，说明当前元素的左边的所有元素都一定小于目标值，因此往左查找不可能找到目标值，往下查找可能找到目标值。）

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
