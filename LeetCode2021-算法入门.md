# LeetCode2021-算法入门

## 二分查找

### 二分查找算法

- 代码：

```python
def search(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        middle = left + (right - left) // 2
        if nums[middle] == target:
            return middle
        if target < nums[middle]:
            right = middle - 1
        else:
            left = middle + 1
    return -1
```

### 第一个错误的版本

- 题目说明：产品的每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。

假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。

你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

- 示例：
输入 n = 5, bad = 4， 输出 4  ——4是第一个错误的版本

- 代码

```py
def firstBadVersion(self, n):
    left, right = 0, n-1
    while left <= right:
        middle = left + (right - left) // 2
        if isBadVersion(middle+1):  # 第几版本是位置信息+1
            right = middle - 1
        else:
            left = middle + 1
    return left+1
```

### 搜索插入位置

- 题目描述：

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

- 示例：

输入: nums = [1,3,5,6], target = 5
输出: 2
输入: nums = [1,3,5,6], target = 7
输出: 4
输入: nums = [1,3,5,6], target = 0
输出: 0

- 代码

```py
def searchInsert(self, nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        middle = left + (right - left) // 2
        if nums[middle] == target:
            return middle
        if target < nums[middle]:
            right = middle - 1
        else:
            left = middle + 1
    
    if nums[left] < target:
        return left + 1     # 插后面
    else:
        return left         # 插左边
```

## 双指针

### 有序数组的平方

- 题目说明：给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序。

- 示例：
输入：nums = [-4,-1,0,3,10]
输出：[0,1,9,16,100]
解释：平方后，数组变为 [16,1,0,9,100]
排序后，数组变为 [0,1,9,16,100]

- 代码

```py
def sortedSquares(self, nums):
    n = len(nums)
    result = [0] * n
    
    # 双指针，nums从两端开始渐小，所以逆序往result里面放即可
    i, j, pos = 0, n - 1, n - 1
    while i <= j:
        if nums[i] ** 2 > nums[j] ** 2:
            result[pos] = nums[i] ** 2
            i += 1
        else:
            result[pos] = nums[j] ** 2
            j -= 1
        pos -= 1
    
    return result
```

- 注意：如果是用排序方法做，时间复杂度O（nlogn），空间复杂度O（1）——根据排序算法而定

### 倒排数组

- 题目说明：给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

- 示例：
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

- 思路：这种case有专属解决方案——可以参考左移动，先左翻转，再又翻转，最后整体翻转。
- 代码：

```py
def rotate(self, nums, k):
    def reverse(i, j):
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
    n = len(nums)
    k %= n
    # 整体翻转，然后左右各自翻转，恰好完美实现
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

### 移动零

- 题目描述：给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

- 示例：

输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

- 代码：

```py
def moveZeroes(self, nums):
    left, right = 0, 0
    # 左指针左边均为非零数；
    # 右指针左边直到左指针处均为零。
    while right < len(nums):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
        right += 1
```

- 思考：虽然想到了这个方案，但如果没注意到`右指针左边直到左指针处均为零`这一点外，写的程序就要多做很多次判断了。

### 两数之和 II - 输入有序数组

- 题目描述：给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 1 开始计数 ，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。

你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

- 示例：

输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

- 代码：

```py
def twoSum(self, numbers, target):
    low, high = 0, len(numbers) - 1
    # O(N) 遍历，利用有序数组，如果和>target，右侧左移，如果和小于target，左侧右移
    # 会不会漏解？不会漏解：因为答案位置是固定的，要么左侧先到答案位置，要么右侧先到答案位置。
    # 如果左侧先到，那么此时total>target，那么一定是右侧左移
    # 如果右侧先到，那么此时total<target，那么一定是右侧右移
    while low < high:
        total = numbers[low] + numbers[high]
        if total == target:
            return [low + 1, high + 1]
        elif total < target:
            low += 1
        else:
            high -= 1

    return [-1, -1]
```

### 反转字符串——秒杀

- 题目描述：略
- 示例：
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]
- 代码：

```py
def reverseString(self, s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s
```

- 代码2：`return s.reverse()`

### 反转字符串中的单词 叁

- 题目描述：略
- 示例:
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"

- 代码：`return " ".join(word[::-1] for word in s.split(" "))`
- 代码2：反转字符串的函数，同样形式封装即可，速度会比代码1慢一些。

### 链表的中间结点——秒杀

- 题目描述：给定一个头结点为 head 的非空单链表，返回链表的中间结点。如果有两个中间结点，则返回第二个中间结点。

- 示例：
输入：[1,2,3,4,5]
输出：此列表中的结点 3

- 代码：

```py
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def middleNode(self, head):
        left = right = head
        while right and right.next:
            left = left.next
            right = right.next.next

        return left
```

### 删除链表的倒数第 N 个结点——秒杀

- 题目描述：给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

- 示例：
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
输入：head = [1,2], n = 2
输出：[2]

- 代码：

```py
def removeNthFromEnd(self, head, n):
    real_head = ListNode(0, next=head)
    left = right = real_head

    for i in range(n+1):    # right 先走n+1步
        right=right.next
    while right:            # left走到要删结点的前一位
        right = right.next
        left = left.next
    
    left.next = left.next.next      # 删去结点

    return real_head.next
```

## 滑动窗口

### 无重复字符的最长子串

- 题目描述：给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
- 示例：
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

- 代码：

```py
def lengthOfLongestSubstring(self, s):
    left = right = 0
    content = set()
    max_len = 0
    while right < len(s):
        if s[right] in content:
            content.remove(s[left])
            left += 1
        else:
            content.add(s[right])
            right += 1
            max_len = max(len(content), max_len)
    return max_len
```

- 优化：注意此时，left可能需要移动多次才能到新的位置。优化思路：记录字符出现的下标

```py
def lengthOfLongestSubstring(self, s):
    start, result, c_dict = -1, 0, {}
    for i, c in enumerate(s):
        if c in c_dict and c_dict[c] > start:  # 字符c在字典中 且 上次出现的下标大于当前长度的起始下标
            start = c_dict[c]
            c_dict[c] = i
        else:
            c_dict[c] = i
            result = max(result, i-start)
    return result
```

### 字符串的排列

- 题目描述：给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。
- 示例
输入：s1 = "ab" s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一 ("ba").

- 代码：

```py
# 比较排列实际为比较字符数量
def checkInclusion(self, s1, s2):
    if len(s1) > len(s2):
        return False
    dic1 = [0]*26
    dic2 = [0]*26
    for i in range(len(s1)):
        dic1[ord(s1[i])-ord('a')] += 1
        dic2[ord(s2[i])-ord('a')] += 1
    if dic1 == dic2:
        return True

    # 滑动窗口，左进一个，右出一个
    for i in range(len(s1),len(s2)):
        dic2[ord(s2[i-len(s1)])-ord('a')] -= 1
        dic2[ord(s2[i])-ord('a')] += 1
        if dic1 == dic2:
            return True
    return False
```

- 优化：使用dif表示一进一出的差距（注：理论上是优化的，不需要每次比较整个列表，但LeetCode用时反而增加了）

```py
def checkInclusion(self, s1, s2):
    if len(s1) > len(s2):
        return False
    lst = [0]*26
    for i in range(len(s1)):
        lst[ord(s1[i])-ord('a')] += 1
        lst[ord(s2[i])-ord('a')] -= 1
    
    dif = sum(map(bool, lst))
    if dif == 0:
        return True
    for i in range(len(s1),len(s2)):
        x = ord(s2[i-len(s1)])-ord('a')
        y = ord(s2[i])-ord('a')
        if x==y:
            continue
        lst[x] += 1
        lst[y] -= 1
        if lst[x] == 1:
            dif += 1
        if lst[x] == 0:
            dif -=1
        if lst[y] == -1:
            dif += 1
        if lst[y] == 0:
            dif -=1
        if dif == 0:
            return True
    return False
```

## 深度优先搜索

### 图像渲染

- 题目描述：

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。

给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。

为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。

最后返回经过上色渲染后的图像。

- 示例：

输入:
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析:
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。

- 代码：
  
```py
n, m = len(image), len(image[0])
currColor = image[sr][sc]

def dfs(x, y):
    if image[x][y] == currColor:
        image[x][y] = newColor
        for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= mx < n and 0 <= my < m and image[mx][my] == currColor:
                dfs(mx, my)

if currColor != newColor:
    dfs(sr, sc)
return image
```

### 岛屿最大面积

- 题目说明：给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

- 示例：
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1 。

- 代码：

```py
def maxAreaOfIsland(self, grid):
    # 不需要标记矩阵，因为我们可以直接改grid
    def bfs(x, y, count):
        count = 1
        grid[x][y] = 0
        for nx, ny in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]:
            if 0 <= nx < n and 0<= ny < m and grid[nx][ny]==1:
                count += bfs(nx, ny, count)
        return count
        
    n, m = len(grid), len(grid[0])
    max_count = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1:
                max_count = max(bfs(i, j, 0), max_count) 
    
    return max_count
```

### 合并二叉树

- 题目说明：给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。

你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

- 示例：

```bash
输入: 
    Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
         3
        / \
       4   5
      / \   \ 
     5   4   7
```

- 代码：

```py
def mergeTrees(self, node1, node2):
    if not node1:       # node1不存在，返回node2；两个都不存在，返回None
        return node2
    if not node2:
        return node1
    
    # 只有两个都存在的时候才需要新建结点，否则直接用原来的结点就好了
    merged = TreeNode(node1.val + node2.val)
    merged.left = self.mergeTrees(node1.left, node2.left)
    merged.right = self.mergeTrees(node1.right, node2.right)
    return merged
```

### 填充每个节点的下一个右侧节点指针

- 题目描述：给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

```c
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

- 示例：

输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。

- 代码：

```py
def connect(self, root: 'Node') -> 'Node':
    if not root:
        return root
    Q = collections.deque([root])

    while Q:
        size = len(Q)
        # 每一层的长度遍历，那就不需要记录专门的count
        for i in range(size):
            node = Q.popleft()
            if i < size - 1:
                node.next = Q[0]
            if node.left:
                Q.append(node.left)
            if node.right:
                Q.append(node.right)

    return root
```

- 优化：对于本题，有可以优化空间的思路：如果是一个结点的左右结点，那么左节点next就是右节点；如果是不同结点的左右结点，那么可以通过两个父节点的关系，找到前一个父节点的右节点，next到下一个父节点的左节点。

```py
def connect(self, root: 'Node') -> 'Node':
    if not root:
        return root
    leftmost = root

    while leftmost.left:
        head = leftmost
        while head:
            # 第一种连接：一个父节点的左右子结点
            head.left.next = head.right

            # 第二种连接：前一个父节点的右子结点，连到下一个父节点的左子结点
            if head.next:
                head.right.next = head.next.left
            
            head = head.next
        leftmost = leftmost.left

    return root 
```

## 广度优先搜索

### 01矩阵

- 题目描述：给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

- 示例：
输入：mat = [[0,0,0],[0,1,0],[1,1,1]]
输出：[[0,0,0],[0,1,0],[1,2,1]]

- 代码

```py
def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
    m, n = len(matrix), len(matrix[0])
    dist = [[0] * n for _ in range(m)]
    zeroes_pos = [(i, j) for i in range(m) for j in range(n) if matrix[i][j] == 0]
    # 将所有的 0 添加进初始队列中，构建一个“零集合”，由零集合开始广度优先搜索
    q = collections.deque(zeroes_pos)
    seen = set(zeroes_pos)

    # 广度优先搜索
    while q:
        i, j = q.popleft()
        for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
                dist[ni][nj] = dist[i][j] + 1
                q.append((ni, nj))
                seen.add((ni, nj))
    
    return dist
```

- 注：此题还有dp方法可解，有空见[leetcode](https://leetcode-cn.com/problems/01-matrix/solution/)