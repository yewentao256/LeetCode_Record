# LeetCode Top 150 Interview

- [LeetCode Top 150 Interview](#leetcode-top-150-interview)
  - [Easy](#easy)
    - [Number](#number)
      - [Reverse Bits](#reverse-bits)
      - [Palindrome Number](#palindrome-number)
    - [Tree](#tree)
      - [Minimum Absolute Difference in BST](#minimum-absolute-difference-in-bst)
    - [Double Pointer](#double-pointer)
      - [Merge Sorted Array](#merge-sorted-array)
      - [Remove Element](#remove-element)
    - [String](#string)
      - [Add Binary](#add-binary)
    - [Array](#array)
      - [Word Pattern](#word-pattern)
      - [Majority Element](#majority-element)
      - [Best Time to Buy and Sell Stock](#best-time-to-buy-and-sell-stock)
  - [Medium](#medium)
    - [Double Pointer](#double-pointer-1)
      - [Container With Most Water](#container-with-most-water)
      - [Remove Duplicates from Sorted Array II](#remove-duplicates-from-sorted-array-ii)
    - [Greedy](#greedy)
      - [Jump Game II](#jump-game-ii)
    - [Graph(DFS)](#graphdfs)
      - [Word Search](#word-search)
      - [Course Schedule II](#course-schedule-ii)
      - [Course Schedule](#course-schedule)
      - [Evaluation Division](#evaluation-division)
      - [Clone Graph](#clone-graph)
      - [Combinations](#combinations)
    - [Array](#array-1)
      - [Insert Interval](#insert-interval)
      - [Longest Increasing Subsequence](#longest-increasing-subsequence)
      - [Minimum Genetic Mutation](#minimum-genetic-mutation)
      - [Find the Index of the First Occurrence in a String(KMP)](#find-the-index-of-the-first-occurrence-in-a-stringkmp)
      - [Product of Array Except Self](#product-of-array-except-self)
      - [H-Index](#h-index)
      - [Insert Delete GetRandom O(1)](#insert-delete-getrandom-o1)
      - [Valid Sudoku](#valid-sudoku)
    - [Tree](#tree-1)
      - [Construct Quad Tree](#construct-quad-tree)
      - [Flatten Binary Tree to Linked List](#flatten-binary-tree-to-linked-list)
      - [Implement Trie (Prefix Tree)](#implement-trie-prefix-tree)
      - [Design Add and Search Words Data Structure (Prefix Tree)](#design-add-and-search-words-data-structure-prefix-tree)
    - [Dynamic Programming](#dynamic-programming)
      - [Edit Distance](#edit-distance)
      - [Interleaving String](#interleaving-string)
      - [Longest Palindromic Substring](#longest-palindromic-substring)
      - [Coin Change](#coin-change)
      - [Word Break](#word-break)
      - [Triangle](#triangle)
      - [Maximum Subarray](#maximum-subarray)
      - [Maximum Sum Circular Subarray](#maximum-sum-circular-subarray)
    - [Dict](#dict)
      - [Group Anagrams](#group-anagrams)
    - [Heap](#heap)
      - [Kth Largest Element in an Array](#kth-largest-element-in-an-array)
  - [Hard](#hard)
    - [Graph](#graph)
      - [Word Ladder](#word-ladder)
    - [String](#string-1)
      - [Substring with Concatenation of All Words](#substring-with-concatenation-of-all-words)
    - [Heap](#heap-1)
      - [IPO](#ipo)
    - [Array](#array-2)
      - [Candy](#candy)
      - [Trap Water](#trap-water)

link: [https://leetcode.cn/studyplan/top-interview-150/]

Only questions that I can't pass are recorded

## Easy

### Number

#### Reverse Bits

Q: Reverse bits of a given 32 bits unsigned integer.

Eg:

```bash
Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
```

Solution: Swap the bits gradually, Time: `O(1)` Space: `O(1)`

```py
def reverseBits(self, n: int) -> int:
    m1 = 0x55555555     # 01010101010101010101010101010101
    m2 = 0x33333333     # 00110011001100110011001100110011
    m4 = 0x0f0f0f0f     # 00001111000011110000111100001111
    m8 = 0x00ff00ff     # 00000000111111110000000011111111

    # Step 1: Swap odd and even bits
    n = ((n >> 1) & m1) | ((n & m1) << 1)

    # Step 2: Swap consecutive pairs
    n = ((n >> 2) & m2) | ((n & m2) << 2)

    # Step 3: Swap nibbles (4 bits)
    n = ((n >> 4) & m4) | ((n & m4) << 4)

    # Step 4: Swap bytes
    n = ((n >> 8) & m8) | ((n & m8) << 8)

    # Step 5: Swap 2-byte long pairs
    n = (n >> 16) | (n << 16)

    return n & 0xFFFFFFFF  # Ensure n is within 32-bit bounds
```

#### Palindrome Number

Q: Given an integer x, return true if x is a palindrome, and false otherwise.

Eg:

```bash
Input: x = -121
Output: false
Input: x = 121
Output: true
Input: x = 10
Output: false
```

Solution: reverted half of the number, then compare. Time: `O(logN)`, Space: `O(1)`

```py
def isPalindrome(x: int) -> bool:
    # x < 0, not satisfied
    # the last number is 0, not satisfied if not 0 itself
    if x < 0 or (x % 10== 0 and x != 0):
        return False

    # eg: 1221, try to get "21" and reverted
    # first round: reverted_number = 1
    # second round: reverted number = 12
    reverted_number = 0
    while x > reverted_number:
        reverted_number = reverted_number * 10 + x % 10
        x = x // 10

    # x == (reverted_number // 10) is for case like 12321
    # x = 12 and reverted_number = 123
    return x == reverted_number or x == (reverted_number // 10)
```

### Tree

#### Minimum Absolute Difference in BST

Q: Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.

Eg:

```bash
        4
      /   \
    2       6
  /    \
1       3

output: 1
```

Solution: BST, So **In-Order Traversal**, Time: `O(N)`, Space: `O(logN)` (Maximum `O(N)` when linked list)

```py
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        self.prev = -float('inf')
        self.min_diff = float('inf')
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)

            # in-order
            self.min_diff = min(self.min_diff, node.val - self.prev)
            self.prev = node.val

            inorder(node.right)

        inorder(root)
        return self.min_diff
```

### Double Pointer

#### Merge Sorted Array

Q: You are given two integer arrays `nums1` and `nums2`, sorted in **non-decreasing** order, and two integers `m` and `n`, representing the number of elements in `nums1` and `nums2` respectively. Merge `nums1` and `nums2` into a single array sorted in **non-decreasing** order.

Note: In-place operation

Eg:

```bash
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
```

Solution: Double pointer, **starts from the tail**, Time: `O(N)`, Space: `O(1)`

```py
def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    i, j, index = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[index] = nums1[i]
            i -= 1
        else:
            nums1[index] = nums2[j]
            j -= 1
        index -= 1

```

#### Remove Element

Q: Given an integer array `nums` and an integer `val`, remove all occurrences of `val` in `nums` in-place. Then return the number of elements in nums which are not equal to val.

Eg:

```bash
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Note: the order is not cared
```

Solution: Double pointer, **one from the beginning, one from the end**, Time: `O(N)`, Space: `O(1)`

```py
def removeElement(nums: List[int], val: int) -> int:
    # why j = len(nums) instead of len(nums) - 1? To handle the case when len(nums) = 1
    i, j = 0, len(nums)
    while(i < j):
        if nums[i] == val:
            nums[i] = nums[j-1]
            # why not i += 1 here? nums[j - 1] may still equal to val
            # try to move i less, ensure that elements below i are suitable
            j -= 1
        else:
            i += 1
    return i
```

### String

#### Add Binary

Q: Given two binary strings `a` and `b`, return their sum as a binary string.

Eg:

```bash
Input: a = "11", b = "1"
Output: "100"
Input: a = "1010", b = "1011"
Output: "10101"
```

Solution: Pad with `"0"`, then iterate from the lasting bit, finally add the carry bit. Time: `O(N)`, Space: `O(N)`

```py
def addBinary(a: str, b: str) -> str:
    max_len = max(len(a), len(b))

    # Pad the shorter string with zeros
    a = a.zfill(max_len)
    b = b.zfill(max_len)

    result = []
    carry = 0

    # Add from the last digit to the first
    for i in range(max_len - 1, -1, -1):
        sum = carry
        sum += 1 if a[i] == '1' else 0
        sum += 1 if b[i] == '1' else 0
        
        # Append the result of (sum % 2) and calculate the new carry
        result.append(str(sum % 2))
        carry = sum // 2

    # If there is a carry left after the last addition, add it to the result
    if carry != 0:
        result.append(str(carry))

    # Since the result array is built backwards, we need to reverse it
    result.reverse()

    return ''.join(result)
```

### Array

#### Word Pattern

Q: Given a `pattern` and a string `s`, find if s follows the same pattern.

Here **follow** means a full match, such that there is a bijection between a letter in `pattern` and a **non-empty** word in `s`.

Eg:

```bash
Input: pattern = "abba", s = "dog cat cat dog"
Output: true
Input: pattern = "abba", s = "dog cat cat fish"
Output: false
Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false
```

Solution: Hashmap, Time: `O(N)`, Space: `O(N)`. Note that this is a **bijection**, so `pattern ="abba"` and `s = "dog dog dog dog"` is not allowed

```py
def wordPattern(pattern: str, s: str) -> bool:
    lst = s.split(' ')
    dic = {}
    reverse_dic = {}
    if len(pattern) != len(lst):
        return False
    
    for p, word in zip(pattern, lst):
        if (p in dic and dic[p] != word) or (word in reverse_dic and reverse_dic[word] != p):
            return False
        dic[p] = word
        reverse_dic[word] = p
    return True
```

#### Majority Element

Q: Given an array `nums` of size `n`, return the majority element. You may assume that the majority element always exists in the array.

Eg:

```bash
Input: nums = [3,2,3]
Output: 3
Input: nums = [2,2,1,1,1,2,2]
Output: 2
```

Solution: **Vote** (Boyer-Moore), Time: `O(N)`, Space: `O(1)`

```py
def majorityElement(nums: List[int]) -> int:
    count = 0
    for num in nums:
        if count == 0:
            # Note: count is always >= 0
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate
```

#### Best Time to Buy and Sell Stock

Q: You are given an array `prices` where `prices[i]` is the price of a given stock on the ith day. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Eg:

```bash
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Input: prices = [7,6,4,3,1]
Output: 0
```

Solution: traverse once -- we can't only find the minimum value to buy, we have to consider the max_profit we now can get. Time: `O(N)`, Space: `O(1)`

```py
def maxProfit(prices: List[int]) -> int:
    # key: find the best place to buy and to sell
    # key2: record the max_profit all the time
    min_index = -1
    min_value, max_value = 10**4, 0
    max_profit = 0
    for i in range(len(prices)):
        if prices[i] < min_value:
            if max_value - min_value > max_profit:
                max_profit = max_value - min_value
            min_value, min_index = prices[i], i
            max_value = min_value   # reset max_value
        elif i > min_index and prices[i] > max_value:
            max_value = prices[i]
    return max(max_profit, max_value - min_value)
```

## Medium

### Double Pointer

#### Container With Most Water

Q and Eg: [https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-interview-150]

Solution: Double pointer + greedy, Time: `O(N)`, Space: `O(1)`

```py
def maxArea(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    result = 0
    while left < right:
        area = min(height[left], height[right]) * (right - left)
        result = max(area, result)
        # Why greedy here works?
        # Prove: we should always move the smaller one
        # originally, area = min(height[left], height[right]) * (right - left)
        # if height[right] > height[left] and we want to move right
        # new_area = min(height[left], height[right']) * (right' - left)
        # we have 1: min(height[left], height[right']) <= min(height[left], height[right]) forever, (if height[right'] > height[right], min(..) = height[left])
        # we have 2: (right' - left) < (right - left) forever
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return result
```

#### Remove Duplicates from Sorted Array II

Q: Given an integer array `nums` sorted in **non-decreasing** order, remove some duplicates **in-place** such that each unique element appears at most twice. The **relative order** of the elements should be kept the same.

Eg:

```bash
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
# Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
```

Solution: slow and fast pointer, Time: `O(N)`, Space: `O(1)`

```python
def removeDuplicates(nums: List[int]) -> int:
    # key: judge by key-2, if nums[i-2] == nums[j], nums[i] must == nums[j]
    # key2: nums[i] should be covered each time when i moves
    if len(nums) <= 2:
        return len(nums)
    i, j = 2, 2
    while(j < len(nums)):
        if nums[i - 2] == nums[j]:
            # find the duplicate case, keep going and find
            j += 1
        else:
            # replace the duplicate
            nums[i] = nums[j]
            i += 1
            j += 1
        
    return i
```

### Greedy

#### Jump Game II

Q: You are given a 0-indexed array of integers nums of length `n`. You are initially positioned at `nums[0]`. Each element `nums[i]` represents the maximum length of a forward jump from index i. Return the minimum number of jumps to reach `nums[n - 1]`. The test cases are generated such that you can reach `nums[n - 1]`.

Eg:

```bash
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

Input: nums = [2,3,0,1,4]
Output: 2
```

Solution: "Global" Greedy, Time: `O(N)`, Space: `O(1)`

```py
def jump(nums: List[int]) -> int:
    # "global" greedy: traverse each index, get the max_index
    # if we arrive at the max_index for a jump, count += 1
    max_index = 0  # Currently the max index we can jump
    end = 0        # The max index we can jump at last time
    count = 0
    # why len(nums) - 1? the last index doesn't need to execute
    for i in range(len(nums) - 1):
        if max_index >= i:
            max_index = max(max_index, i + nums[i])
            if i == end:
                end = max_index
                count += 1
    return count
```

### Graph(DFS)

#### Word Search

Q and eg: See [leetcode](https://leetcode.cn/problems/word-search/description/?envType=study-plan-v2&envId=top-interview-150)

Solution: DFS + traceback. Time: `O(N*M*3^k)`, Space: `O(k)`, k means the length of word.

```py
from collections import Counter
def exist(board: List[List[str]], word: str) -> bool:
    m, n = len(board), len(board[0])

    # Optimize: Word count, if not enough characters for word, return False
    board_count = Counter(char for row in board for char in row)
    word_count = Counter(word)
    for char in word_count:
        if board_count.get(char, 0) < word_count[char]:
            return False

    # Optimize: if too much first character, reverse the word
    if board_count[word[0]] > board_count[word[-1]]:
        word = word[::-1]

    def dfs(i: int, j: int, index: int) -> bool:
        if index == len(word):
            return True
        # check border and character match 
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[index]:
            return False

        # mark current board as visited(temp)
        temp = board[i][j]
        board[i][j] = '#'

        # search in four directions
        found = (dfs(i+1, j, index+1) or
                dfs(i-1, j, index+1) or
                dfs(i, j+1, index+1) or
                dfs(i, j-1, index+1))

        # restore the temp character
        board[i][j] = temp

        return found

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True

    return False
```

#### Course Schedule II

Q: There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

Eg:

```bash
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].

Input: numCourses = 1, prerequisites = []
Output: [0]
```

Solution: DFS + check circle in DAG

```py
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # Main idea: find a path in DAG, ensuring no circle
    # Solution: DFS(path + check if there is a circle)

    # build graph
    graph = [[] for _ in range(numCourses)]
    for dest, src in prerequisites:
        graph[src].append(dest)

    # dfs
    # Note: we can't deal with a node and add to path immediately
    # eg: 1 -> 3, 2-> 3. If immediately, 3 will be added
    # The correct solution is to add the node later, so 3 will be added in first index
    # then make it reversed
    path = []
    visited = [0] * numCourses  # 0 for unvisited, 1 for visiting, 2 for visited
    def dfs(course: int) -> bool:
        if visited[course] == 2:
            return True
        if visited[course] == 1:
            # a circle
            return False
        visited[course] = 1
        for next_course in graph[course]:
            if not dfs(next_course):
                return False
        visited[course] = 2
        path.append(course)
        return True

    for start_course in range(numCourses):
        if not dfs(start_course):
            return []
    
    return path[::-1]
```

#### Course Schedule

Q: There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

Eg:

```bash
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
```

Solution: Check circle in a graph(DAG)--DFS

```py
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    # Time: O(V+E), Space: O(V+E)
    graph = defaultdict(list)
    for dest, src in prerequisites:
        graph[src].append(dest)

    # 0: unvisted, 1: visiting. 2: visited
    visited = [0] * numCourses
    
    def dfs(course: int) -> bool:
        if visited[course] == 1:
            # circle
            return False
        if visited[course] == 2:
            # visited, return earlier
            return True
        
        visited[course] = 1
        for next_course in graph[course]:
            if not dfs(next_course):
                return False
        visited[course] = 2
        return True

    # Note: we will visit all of the courses, so as long as no circle, we can study all of the courses
    # In other words, isolated node can be counted as studied directly.
    for course in range(numCourses):
        if not dfs(course):
            return False

    return True
```

#### Evaluation Division

Q: You are given an array of variable pairs equations and an array of real numbers values, where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the answer for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Eg:

```bash
Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
Explanation: 
Given: a / b = 2.0, b / c = 3.0
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? 
return: [6.0, 0.5, -1.0, 1.0, -1.0 ]
note: x is undefined => -1.0

Input: equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
Output: [3.75000,0.40000,5.00000,0.20000]

Input: equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
Output: [0.50000,2.00000,-1.00000,-1.00000]
```

Solution: Graph(`dict[str, list]`), Time: `O(Q * E)`(queries * equations), Space: O(V + E) (variables + equations)

```py
def calcEquation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    graph = defaultdict(list)
    for (A, B), value in zip(equations, values):
        graph[A].append((B, value))
        graph[B].append((A, 1/value))
    
    def dfs(current: str, target: str, accumulated: float, visited:set) -> float:
        if current == target:
            return accumulated
        visited.add(current)
        for neighbor, value in graph[current]:
            if neighbor in visited:
                continue
            result = dfs(neighbor, target, accumulated * value, visited)
            if result != -1.0:
                return result
        return -1
    results = []
    for C, D in queries:
        if C not in graph or D not in graph:
            results.append(-1)
        elif C == D:
            results.append(1)
        else:
            results.append(dfs(C, D, 1.0, set()))
    return results        
```

#### Clone Graph

Questions and Example: Details at [LeetCode](https://leetcode.cn/problems/clone-graph/)

Solution: DFS + record the map

```py
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # DFS while recording the cloned nodes
        # Time: O(N), Space: O(N)
        if not node:
            return None
        
        # Note: We can't record the edge as visited, since we may create multiple repeated nodes
        cloned_nodes = {}
        
        def dfs(original_node: Node) -> Node:
            if original_node in cloned_nodes:
                return cloned_nodes[original_node]
            
            cloned_node = Node(original_node.val)
            cloned_nodes[original_node] = cloned_node
            
            for neighbor in original_node.neighbors:
                cloned_node.neighbors.append(dfs(neighbor))
                
            return cloned_node
        
        return dfs(node)
```

#### Combinations

Q: Given two integers `n` and `k`, return all possible combinations of `k` numbers chosen from the range `[1, n]`.

Eg:

```bash
Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
Explanation: There are 4 choose 2 = 6 total combinations.
Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.

Input: n = 1, k = 1
Output: [[1]]
Explanation: There is 1 choose 1 = 1 total combination.
```

Solution: DFS + trace back, Time: `O(C(n, k) * k)` (count of results * k times). Space: `O(k + k)`(stack trace + temp l)

```py
def combine(n: int, k: int) -> List[List[int]]:
    results = []
    l = []
    def dfs(start: int, k: int) -> None:
        if not k:
            results.append(l.copy())
            return
        for num in range(start, n + 1):
            l.append(num)
            dfs(num + 1, k - 1)
            l.pop()
    dfs(1, k)
    return results
```

### Array

#### Insert Interval

Q: Insert a `newInterval` into intervals (which is sorted by start_i in ascending order.)

Eg:

```bash
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
```

Solution: Binary search(conservative) + iterate

```py
from bisect import bisect_left

def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
    # Time: O(N) but with the help of bisect, it could be closer to O(logN)
    # Space: O(1) if results are not counted
    results = []

    # Binary search to find the correct insertion position
    i = bisect_left(intervals, newInterval[0], key=lambda x: x[0])
    if i > 0:
        # conservative strategy
        i -= 1

    # Add all intervals before newInterval (those that end before the newInterval starts)
    results.extend(intervals[:i])
    if i < len(intervals) and intervals[i][1] < newInterval[0]:
        results.append(intervals[i])
        i += 1

    # Merge the overlapping intervals with newInterval
    while i < len(intervals) and intervals[i][0] <= newInterval[1]:
        # this is needed like [1, 3][6, 9] insert with [2,5], i start with 0 
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    # Add the merged newInterval and the rest
    results.append(newInterval)
    results.extend(intervals[i:])
    return results
```

#### Longest Increasing Subsequence

Q: Given an integer array nums, return the length of the longest strictly increasing subsequence.

Eg:

```bash
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Input: nums = [0,1,0,3,2,3]
Output: 4
Input: nums = [7,7,7,7,7,7,7]
Output: 1
```

Solution1: Dynamic Programming. Time: `O(N^2)`, Space: `O(N)`

```py
def lengthOfLIS(nums: List[int]) -> int:
    # dp[i] = to the index i, the length of the longest sequence
    # dp[0] = 1
    # dp[i] = max(dp[j]) + 1  (j from 0 ~ i)
    dp = [0] * len(nums)
    dp[0] = 1
    for i in range(1, len(nums)):
        
        for j in range(0, i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i] - 1, dp[j]) + 1
        if not dp[i]:
            dp[i] = 1
    return max(dp)
```

Solution2: Greed + Bisect. Time: `O(N * logN)`, Space: `O(N)`

```py
from typing import List

def my_bisect_left(array: List[int], val: int) -> int:
    left, right = 0, len(array)
    while left < right:
        mid = (left + right) // 2
        if array[mid] < val:
            # mid has been used, left should + 1
            left = mid + 1
        else:
            # right means the upper index you can not visit, no needs to -1
            right = mid
    return left

def lengthOfLIS(nums: List[int]) -> int:
    # tails[i] means when length == i+1, LIS' last element
    # So len(tails[i]) == len(LIS)
    # eg: [10,9,2,5,3,7,101,18,1] we have tails = [1, 3, 7, 18]
    # meaning that we have the LIS length == 4, the last element is 18
    # for tails[2] = 3, means we can find the sequence [2,3] for the brightest future
    # This is a kind of greedy solution.
    
    tails = []
    
    for num in nums:
        index = my_bisect_left(tails, num)
        if index == len(tails):
            # last element < num, means we can expand our LIS
            tails.append(num)
        else:
            # we can update our LIS, to build a better one, ending with the num
            # eg, tail = [1], means for length == 1's sequence, use [1] is the best
            # eg2, tail = [1,3,7,18], tail[2] = 7 means for length == 3's sequence, use [2, 3, 7](ending with 7) is the best 
            tails[index] = num
    return len(tails)
```

#### Minimum Genetic Mutation

Q: A gene string can be represented by an 8-character long string, with choices from `'A'`, `'C'`, `'G'`, and `'T'`.

Suppose we need to investigate a mutation from a gene string `startGene` to a gene string `endGene` where one mutation is defined as one single character changed in the gene string.

For example, `"AACCGGTT" --> "AACCGGTA"` is one mutation.
There is also a gene `bank` that records all the valid gene mutations. A gene must be in `bank` to make it a valid gene string.

Given the two gene strings `startGene` and `endGene` and the gene `bank`, return the minimum number of mutations needed to mutate from `startGene` to `endGene`. If there is no such a mutation, return `-1`.

Eg:

```bash
Input: startGene = "AACCGGTT", endGene = "AACCGGTA", bank = ["AACCGGTA"]
Output: 1
Input: startGene = "AACCGGTT", endGene = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
Output: 2
```

Solution: Brutal BFS

```py
def minMutation(startGene: str, endGene: str, bank: List[str]) -> int:
    # Time: O(3 * 8 * N), N means the number of strings in bank
    # Space: O(8 * N)
    bank_set = set(bank)
    if endGene not in bank_set:
        return -1
    
    queue = deque([(startGene, 0)])
    visited = set([startGene])

    while queue:
        current_gene, mutation_count = queue.popleft()

        # for each char, try all possible mutation
        for i in range(len(current_gene)):
            for char in "ACGT":
                if char != current_gene[i]:
                    mutated_gene = current_gene[:i] + char + current_gene[i+1:]
                    if mutated_gene == endGene:
                        return mutation_count + 1
                    if mutated_gene in bank_set and mutated_gene not in visited:
                        visited.add(mutated_gene)
                        queue.append((mutated_gene, mutation_count + 1))
    return -1
```

#### Find the Index of the First Occurrence in a String(KMP)

Q: Given two strings `needle` and `haystack`, return the index of the first occurrence of `needle` in `haystack`, or `-1` if `needle` is not part of `haystack`.

Eg:

```bash
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.

Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
```

Solution: KMP, Time: `O(N + M)`, Space: `O(M)`. The key point of KMP is: The `i` in haystack never goes back.

```py
def build_next(pattern: str) -> list[int]:
    next = [0]
    length = 0      # the common pre/suffix length
    i = 1
    while i < len(pattern):
        if pattern[length] == pattern[i]:
            # matches
            length += 1
            next.append(length)
            i += 1
        else:
            # doesn't match
            if length == 0:
                # no common pre/suffix now
                next.append(0)
                i += 1
            else:
                # eg:              A  B  A  C  A  B  A  B
                # we've built the [0, 0, 1, 0, 1, 2, 3, ?]
                # when calculating `?`, our length = 3, meaning "A B A"
                # not matches, try back, start from "A B", fetch next[length -1]
                # if you don't understand, consider this is a technique
                length = next[length - 1]
    return next

def kmp_search(string: str, pattern: str) -> int:
    # build a next array for skipping some characters
    # it means the common pre/suffix for the ith character
    # eg: ABABC, we can get [0, 0, 1, 2, 0]
    # eg2: ABACABAB, we can get [0, 0, 1, 0, 1, 2, 3, 2]
    next = build_next(pattern)
    
    i, j = 0, 0
    while i  < len(string):
        if string[i] == pattern[j]:
            # matches, move on
            i += 1
            j += 1
        elif j > 0:
            # doesn't match, skip some characters based on next array
            # eg: find ABABC in ABABABCAA, j = 4 and get new j = 2
            j = next[j - 1]
        else:
            # when j = 0, this is a start
            i += 1
        
        if j == len(pattern):
            # matches all of the pattern
            return i - j
    return -1
```

#### Product of Array Except Self

Q: Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

Eg:

```bash
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

Solution: Maintain the `left_sum` before index i and `right_sum` after index i. Time: `O(N)`, Space: `O(1)`

```py
def productExceptSelf(nums: List[int]) -> List[int]:
    l = len(nums)
    result = [1] * l
    left_sum, right_sum = 1, 1
    for i in range(l):
        j = l - 1 - i
        result[i] *= left_sum
        result[j] *= right_sum
        left_sum *= nums[i]
        right_sum *= nums[j]
    return result
```

#### H-Index

Q: Given an array of integers `citations` where `citations[i]` is the number of citations a researcher received for their ith paper, return the researcher's h-index. The h-index is defined as the maximum value of `h` such that the given researcher has published at least `h` papers that have each been cited at least `h` times.

Eg:

```bash
Input: citations = [3,0,6,1,5]
Output: 3
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively.
Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, their h-index is 3.

Input: citations = [1,3,1]
Output: 1
```

Solution1: sorted and traverse, Time: `O(NlogN)`, Space: `O(N)`

```py
def hIndex(self, citations: List[int]) -> int:
    # key: sorted the citations list in ascending order
    # the later citation must be less than the previous one
    sorted_citation = sorted(citations, reverse = True)
    h = 0; i = 0; n = len(citations)
    while i < n and sorted_citation[i] > h:
        print(i, h, sorted_citation[i])
        h += 1
        i += 1
    return h
```

Solution2: Count(For citation that > len(citations), **reduce it to len(citations)**), Time: `O(N)`, Space: `O(N)`

```py
def hIndex(citations: List[int]) -> int:
    l = len(citations)
    total = 0
    counter = [0] * (l+1)
    # count and reduce
    for c in citations:
        if c >= l:
            counter[l] += 1
        else:
            counter[c] += 1
    
    # start from the tail, the previous one must be less than the later one
    for i in range(l, -1, -1):
        total += counter[i]
        if total >= i:
            return i
    return 0
```

Solution3: **Binary Search The H_index**, Time: `O(N * logN)`, Space: `O(1)`

```py
def hIndex(citations: List[int]) -> int:
    # Key: binary search the H_index
    left, right = 0, len(citations)
    while left < right:
        # +1 to avoid cases like `citations = [1]`
        mid = (left + right + 1) >> 1
        count = 0
        for citation in citations:
            if citation >= mid:
                count += 1
        if count >= mid:
            # H_index is in [mid, right]
            left = mid
        else:
            # H_index is in [0, mid)
            right = mid - 1
    return left
```

#### Insert Delete GetRandom O(1)

Q: Implement the `RandomizedSet` class: `bool insert(int val)` Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise. `bool remove(int val)` Removes an item val from the set if present. Returns true if the item was present, false otherwise. `int getRandom()` Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.

You must implement the functions of the class such that each function works in average `O(1)` time complexity.

Eg:

```bash
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.
```

Idea: We may think about hash map, so the answer looks like:

```py
class RandomizedSet:
    def __init__(self):
        self.dic = {}
    def insert(self, val: int) -> bool:
        if self.dic.get(val):
            return False
        self.dic[val] = 1
        return True
    def remove(self, val: int) -> bool:
        if self.dic.get(val):
            del self.dic[val]
            return True
        return False
    def getRandom(self) -> int:
        return choice(list(self.dic.keys()))
```

But this is not correct since `list(self.dic.keys())` is an `O(N)` operation

Solution: Hash dict + list to fetch random number

```py
from random import choice

class RandomizedSet:

    def __init__(self):
        self.dic = {}   # val: index hash map
        self.lst = []   # available index list

    def insert(self, val: int) -> bool:
        if val in self.dic:
            return False
        self.dic[val] = len(self.lst)
        self.lst.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.dic:
            return False
        # swap with the last element in order for O(1) removal
        last_element, idx_to_remove = self.lst[-1], self.dic[val]
        self.lst[idx_to_remove], self.dic[last_element] = last_element, idx_to_remove
        self.lst.pop()
        del self.dic[val]
        return True

    def getRandom(self) -> int:
        return choice(self.lst)

```

#### Valid Sudoku

Q: Determine if a `9 x 9` Sudoku board is valid. Only the filled cells need to be validated according to the **following rules**:

- Each row must contain the digits 1-9 without repetition.
- Each column must contain the digits 1-9 without repetition.
- Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Eg:

```bash
Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
```

Solution: Iterate once, using hash table to record. Time: `O(N^2)` (`O(1)` since 9*9), Space: `O(N^2)`(`O(1)`):

```py
from collections import defaultdict

def isValidSudoku(board: List[List[str]]) -> bool:
    row_dict = defaultdict(dict)
    column_dict = defaultdict(dict)
    region_dict = defaultdict(dict)

    for r in range(len(board)):
        for c in range(len(board[0])):
            value = board[r][c]
            if value == ".":
                continue
            if row_dict[r].get(value):
                return False
            if column_dict[c].get(value):
                return False
            region_id = 3 * (r // 3) + c // 3
            if region_dict[region_id].get(value):
                return False
            row_dict[r][value] = 1
            column_dict[c][value] = 1
            region_dict[region_id][value] = 1
    return True
```

### Tree

#### Construct Quad Tree

Question and example: Details at [LeetCode](https://leetcode.cn/problems/construct-quad-tree/description/)

Solution: Divide and conquer

```py
def construct(grid: List[List[int]]) -> Node:
    # Time: O(N^2), Space: O(logN)
    # How do we get the time complexity?
    # First, T(N) = a * T(n/b) + f(n), divide to `a` subtasks, f(n) means the cost to manage the subtasks
    # Here our a = 4 and b = 2, f(n) = O(1) = O(N^0) (iterate 4 subnodes)
    # So c = 0 < logb(a) = 2, T(N) = O(n^logb(a)) = O(N^2)
    # Note: if c == logb(a), T(N) = O(n^logb(a) * logN)
    # if c > logb(a), T(N) = f(N), means the main cost is to manage
    # Or, we can consider: We will iterate through all grid[i][i] once, O(N^2)
    n = len(grid)
    
    def build_node(left: int, right: int, top: int, bottom: int) -> Node:
        if left == right - 1 and top == bottom - 1:
            # leaf node
            return Node(grid[top][left], True, None, None, None, None)
        else:
            root_x, root_y = (left + right) // 2, (top + bottom) // 2
            top_left = build_node(left, root_x, top, root_y)
            top_right = build_node(root_x, right, top, root_y)
            bottom_left = build_node(left, root_x, root_y, bottom)
            bottom_right = build_node(root_x, right, root_y, bottom)
            
            all_is_leaf = top_left.isLeaf and top_right.isLeaf and bottom_left.isLeaf and bottom_right.isLeaf
            all_value_equals = top_left.val == top_right.val == bottom_left.val == bottom_right.val
            if all_is_leaf and all_value_equals:
                # All equals, merge leaf nodes
                return Node(top_left.val, True, None, None, None, None)
            else:
                # Not leaf node
                return Node(0, False, top_left, top_right, bottom_left, bottom_right)
    
    return build_node(0, n, 0, n)
```

Solution2: Divide and Conquer + prefix sum

```py
def construct(grid: List[List[int]]) -> Node:
    # Divide and Conquer + prefix sum
    # prefix[i][j] = sum(grid[0][0] ~ grid[i][j]) (whole matrix)
    # Time: O(N^2), Space: O(N^2)
    n = len(grid)
    prefix_sum = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Compute prefix sums
    for i in range(n):
        for j in range(n):
            prefix_sum[i + 1][j + 1] = grid[i][j] + prefix_sum[i][j + 1] + prefix_sum[i + 1][j] - prefix_sum[i][j]
            
    def get_sum(x1, y1, x2, y2):
        return prefix_sum[x2 + 1][y2 + 1] - prefix_sum[x1][y2 + 1] - prefix_sum[x2 + 1][y1] + prefix_sum[x1][y1]
    
    def build_node(x1, y1, x2, y2):
        total = get_sum(x1, y1, x2 - 1, y2 - 1)
        area = (x2 - x1) * (y2 - y1)
        if total == 0:
            # all 0s
            return Node(False, True)
        if total == area:
            # all 1s
            return Node(True, True)
        
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        top_left = build_node(x1, y1, mid_x, mid_y)
        top_right = build_node(x1, mid_y, mid_x, y2)
        bottom_left = build_node(mid_x, y1, x2, mid_y)
        bottom_right = build_node(mid_x, mid_y, x2, y2)
        
        if top_left.isLeaf and top_right.isLeaf and bottom_left.isLeaf and bottom_right.isLeaf:
            if top_left.val == top_right.val == bottom_left.val == bottom_right.val:
                return Node(top_left.val, True)
        
        return Node(True, False, top_left, top_right, bottom_left, bottom_right)
    
    return build_node(0, 0, n, n)
```

#### Flatten Binary Tree to Linked List

Q: Given the `root` of a binary tree, flatten the tree into a "linked list":

- The "linked list" should use the same `TreeNode` class where the `right` child pointer points to the next node in the list and the `left` child pointer is always null.
- The "linked list" should be in the same order as a pre-order traversal of the binary tree.

Eg:

```bash
       1                    1
      /  \                    2
    2      5       =>           3
   /  \     \                    4
  3    4     6                     5
                                    6

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]
Input: root = []
Output: []
Input: root = [0]
Output: [0]
```

Solution: Update in-place, Time: `O(N)`, Space: `O(1)`. Only consider for node(leftmost and rightmost), do not consider as a chain!

```py
def flatten(root: Optional[TreeNode]) -> None:
    # Time: O(N), Space: O(1)
    current = root
    
    while current:
        if current.left:
            # find the rightmost node in the left subtree
            rightmost = current.left
            while rightmost.right:
                rightmost = rightmost.right
            
            # rewrite the connections
            rightmost.right = current.right
            current.right = current.left
            current.left = None
        current = current.right
```

#### Implement Trie (Prefix Tree)

Q: A **trie** (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. Implement the Trie class:

- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string `word` into the trie.
- `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.
- `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.

Note: `word` and `prefix` consist only of lowercase English letters.

Eg:

```bash
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
```

Solution: Construct the Tree using list or dictionary, then move forward

```py
class Trie:
    # Prefix Tree
    # Time complexity: O(1) for __init__, O(len(string)) for the others
    # Space complexity: O(len(string) * 26)

    def __init__(self):
        self.children = [None] * 26    # from 'a' to 'z'
        self.is_end = False            # mark whether the string is finished

    def insert(self, word: str) -> None:
        # insert a word into the prefix tree
        cur_node = self
        for c in word:
            i = ord(c) - ord("a")
            if not cur_node.children[i]:
                cur_node.children[i] = Trie()
            cur_node = cur_node.children[i]
        cur_node.is_end = True

    def search_prefix(self, prefix: str) -> "Trie":
        cur_node = self
        for c in prefix:
            i = ord(c) - ord("a")
            if not cur_node.children[i]:
                return None
            cur_node = cur_node.children[i]
        return cur_node

    def search(self, word: str) -> bool:
        # search a word, return True if the word has been inserted before
        node = self.search_prefix(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        # search prefix, return True if the prefix exists
        return self.search_prefix(prefix) is not None
```

#### Design Add and Search Words Data Structure (Prefix Tree)

Q: Design a data structure that supports adding new words and finding if a string matches any previously added string. Implement the `WordDictionary` class:

- `WordDictionary()` Initializes the object.
- `void addWord(word)` Adds `word` to the data structure, it can be matched later.
- `bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.

Eg:

```bash
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
```

Solution: Prefix Tree (Trie), Time: `O(N)`(including `add` and `search`, since at most 2 `'.'` will be met), Space: `O(26 * N)`

```py
class WordDictionary:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

    def addWord(self, word: str) -> None:
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = WordDictionary()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        def search_in_node(word, node):
            for i, char in enumerate(word):
                if char == '.':
                    # If the current character is '.', check all possible nodes at this level.
                    for child in node.children.values():
                        if search_in_node(word[i + 1:], child):
                            return True
                    return False
                else:
                    if char not in node.children:
                        return False
                    node = node.children[char]
            return node.is_end_of_word
        
        return search_in_node(word, self)
```

### Dynamic Programming

#### Edit Distance

Q: Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

- Insert a character
- Delete a character
- Replace a character

Eg:

```bash
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')
```

Solution: 2D Dynamic programming, Time: O(`N*M`), Space: O(`N*M`)

```py
def minDistance(word1: str, word2: str) -> int:
    # dp[i][j] means changing i characters from word1 to the characters word2, minumum operations
    # dp[0][0] = 0 (empty)
    # dp[0][j] = j, dp[i][0] = i
    # dp[i][j] = 
    #   dp[i-1][j-1] if s[i] == s[j]
    #   1 + min(
    #       dp[i-1][j]       # delete one character
    #       dp[i-1][j-1]     # replace
    #       dp[i][j-1]       # based on to j-1, insert one character
    #   )

    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = i
    for j in range(1, n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])

    return dp[-1][-1]
```

#### Interleaving String

Q: Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an **interleaving** of `s1` and `s2`.

Eg:

```bash
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
Since s3 can be obtained by interleaving s1 and s2, we return true.

Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.

Input: s1 = "", s2 = "", s3 = ""
Output: true
```

Solution: Dynamic Programming. Time: `O(N^2)`, Space: `O(N^2)`

```py
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    # dp[i][j] means to the ith character of s1, jth character of s2, is interleave or not
    # dp[0][0] = 1
    # dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    # dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    # dp[i][j] = (s1[i-1] == s3[i+j-1] and dp[i-1][j]) or (s2[j-1] == s3[i+j-1] and dp[i][j-1])
    n, m = len(s1), len(s2)
    
    # Be careful, according to the definition, len(s1) + len(s2) must equal to len(s3)!
    if n + m != len(s3):
        return False
    dp = [[False for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = (dp[i-1][j] and s1[i - 1] == s3[i + j - 1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    return dp[-1][-1]
```

Solution Optimization: DP with Space Optimized to `O(N)`

```py
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        # Note: this is not easy to understand, since dp[j] can have multiple i, we can do this since we've done the dp[i][j]
        # dp[j] means to the ith character of s1, jth character of s2, is interleave or not
        # dp[0] = True
        # dp[j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        # dp[j] = (s1[i-1] == s3[i+j-1] and dp[j]) or (s2[j-1] == s3[i+j-1] and dp[j-1])
        n, m = len(s1), len(s2)
        
        # Be careful, according to the definition, len(s1) + len(s2) must equal to len(s3)!
        if n + m != len(s3):
            return False
        dp = [False] * (m + 1)
        dp[0] = True
        for j in range(1, m + 1):
            dp[j] = dp[j-1] and s2[j-1] == s3[j-1]
        for i in range(1, n+1):
            dp[0] = dp[0] and s1[i-1] == s3[i-1]
            for j in range(1, m+1):
                dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or (dp[j-1] and s2[j-1] == s3[i+j-1])
        return dp[-1]
```

#### Longest Palindromic Substring

Q: Given a string s, return the longest **palindromic substring** in s.

Eg:

```bash
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Input: s = "cbbd"
Output: "bb"
```

Solution1: Dynamic Programming. Time: `O(N^2)`, Space: `O(N^2)`

```py
def longestPalindrome(s: str) -> str:
    # dp[i][j] means s[i:j+1] is palindrome or not
    # dp[i][i] = 1
    # if two adjacent characters, dp[i][j] = 1 if c1 == c2 else 0
    # >2 characters, dp[i][j] = 1 if dp[i+1][j-1] and s[i] == s[j] else 0
    n = len(s)
    if n < 2:
        return s

    dp = [[0 for _ in range(n)] for _ in range(n)]
    start, max_len = 0, 1
    
    # init
    for i in range(n):
        dp[i][i] = 1
    
    # 2 adjacent characters:
    for i in range(n - 1):
        if s[i] == s[i+1]:
            dp[i][i+1] = 1
            start = i
            max_len = 2
    
    # >2 characters:
    for length in range(3, n+1):
        for i in range(n - length + 1):
            # eg: length = 3, n = 5, i can iterate in range (0, 2)
            j = i + length - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    # here the last one record is the longest
    return s[start: start + max_len]
```

Solution2: Expand around center, Time: `O(N^2)`, Space: `O(1)`

```py
def longestPalindrome(s: str) -> str:
    def expand_around_center(s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    start, end = 0, 0
    for i in range(len(s)):
        len1 = expand_around_center(s, i, i)        # odd
        len2 = expand_around_center(s, i, i + 1)    # even
        max_len = max(len1, len2)
        if max_len > (end - start):
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    return s[start:end + 1]
```

#### Coin Change

Q: You are given an integer array `coins` representing coins of different denominations and an integer amount representing a total `amount` of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`. You may assume that you have an infinite number of each kind of coin.

Eg:

```bash
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Input: coins = [2], amount = 3
Output: -1
Input: coins = [1], amount = 0
Output: 0
```

Solution: Dynamic programming: Time: `O(N * len(coins))`, Space: `O(N)`

```py
def coinChange(coins: List[int], amount: int) -> int:
    # dp[i] means to realize amount i, the minimum count of coins.
    # init: dp[each coin] = 1, dp[i] = float('inf')
    # dp[i] = min(dp[i - coin]) + 1 if dp[i - coin] > 0 else -1
    if not amount:
        return 0
    dp = [float('inf') for _ in range(amount + 1)]
    for i in range(1, amount + 1):
        for coin in coins:
            if i == coin:
                dp[i] = 1
            elif i - coin > 0 and dp[i - coin] != -1:
                dp[i] = min(dp[i], dp[i-coin] + 1)
    return dp[-1] if dp[-1] != float("inf") else -1
```

#### Word Break

Q: Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

Eg:

```bash
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
```

Solution: Dynamic Programming

```py
def wordBreak(s: str, wordDict: List[str]) -> bool:
    # Time: O(N^2), Space: O(N)
    word_set = set(wordDict)

    # dp[i] means to the ith char, whether can be split
    dp = [False] * (len(s) + 1)
    dp[0] = True    # empty string can be split

    # dp[i] = 1 if from arbitrary 0~i-1(j), dp[j] and s[j:i] can be a word
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[-1]
```

#### Triangle

Q: Given a `triangle` array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index `i` on the current row, you may move to either index i or index i + 1 on the next row.

Eg:

```bash
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
```

Solution: 2D dynamic programming

```py
def minimumTotal(triangle: List[List[int]]) -> int:
    # Time: O(N^2), Space: O(N^2), n is the number lines
    # dp[i][j] means to the ith floor, the minimum value towards index j
    # dp[0][0] = triangle[0][0]
    n, m = len(triangle), len(triangle[-1])
    dp = [[0] * m for _ in range(n)]
    dp[0][0] = triangle[0][0]

    # j = 0: dp[i][0] = dp[i-1][0] + triangle[i][0]
    # j = i: dp[i][i] = dp[i-1][i-1] + triangle[i][i]
    # other: dp[i][j] = traiangle[i][j] + min(dp[i-1][j-1] , dp[i-1][j])
    for i in range(1, n):
        dp[i][0] = triangle[i][0] + dp[i-1][0]
        for j in range(1, i):
            dp[i][j] = triangle[i][j] + min(dp[i-1][j-1] , dp[i-1][j])
        # for the last element, there is only one path
        dp[i][i] = dp[i-1][i-1] + triangle[i][i]
    
    return min(dp[-1])
```

Solution2: Optimize the space complexity from bottom to up

```py
def minimumTotal(triangle: List[List[int]]) -> int:
    # Time: O(N^2), Space: O(N), n is the number lines
    # dp[i] means the minimum path sum from bottom to element i in the current row.
    # Note: NOT the minimum sum for the ith row
    # dp = triangle[-1] for initialization
    n = len(triangle)
    dp = triangle[-1][:]
    # dp = triangle[-1]     Normally we don't update the original array, but this works!

    # Bottom-up calculation from the second last row to the first row
    # dp[j] = triangle[i][j] + min(dp[j+1], dp[j])
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[j] = triangle[i][j] + min(dp[j+1], dp[j])

    return dp[0]
```

Solution3: Use the `triangle` itself

```py
def minimumTotal(triangle: List[List[int]]) -> int:
    triangle.reverse()

    # Dynamic programming from the second row (bottom-up)
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            # Update the current element by adding the minimum of the two adjacent
            # elements from the previous row
            triangle[i][j] += min(triangle[i-1][j], triangle[i-1][j+1])

    # The top element now contains the minimum path sum from top to bottom
    return triangle[-1][0]
```

#### Maximum Subarray

Q: Given an integer array nums, find the subarray with the largest sum, and return its sum.

Eg:

```bash
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
```

Solution: Dynamic Programming / Kadane algorithm. Time: O(N), Space: O(1)

```py
def maxSubArray(nums: List[int]) -> int:
    # Dynamic Programing; Kadane
    # f(i) = max(f(i-1) + nums[i], nums[i])
    # f(0) = nums[0]
    current_sum = max_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

#### Maximum Sum Circular Subarray

Q: Given a **circular integer array** `nums` of length `n`, return the maximum possible sum of a non-empty **subarray** of `nums`.

A circular array means the end of the array connects to the beginning of the array. Formally, the next element of `nums[i]` is `nums[(i + 1) % n]` and the previous element of `nums[i]` is `nums[(i - 1 + n) % n]`.

A **subarray** may only include each element of the fixed buffer `nums` at most once.

Eg:

```bash
Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.

Input: nums = [5,-3,5]
Output: 10
Input: nums = [-3,-2,-3]
Output: -2
```

Solution: Kadane's algorithm + inverted nums. Time: `O(N)`, Space: `O(1)`

```py
def maxSubarraySumCircular(nums: List[int]) -> int:
    # Note: the sum of `nums[0:i]` and `nums[j:n]` equals to sum - `nums[i:j]`
    def kadane(arr: List[int]) -> List[int]:
        current_sum = max_sum = arr[0]
        for x in arr[1:]:
            current_sum = max(x, current_sum + x)
            max_sum = max(max_sum, current_sum)
        return max_sum

    total_sum = sum(nums)
    max_kadane = kadane(nums)
    
    # To find the minimum subarray sum, invert the values and apply Kadane's algorithm
    inverted_nums = [-x for x in nums]
    min_kadane = kadane(inverted_nums)
    max_circular = total_sum + min_kadane

    # If max_circular is zero, it means all elements are negative
    if max_circular == 0:
        return max_kadane
    
    return max(max_kadane, max_circular)
```

### Dict

#### Group Anagrams

Q: Given an array of strings `strs`, group the **anagrams** together. You can return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Eg:

```bash
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Input: strs = [""]
Output: [[""]]
Input: strs = ["a"]
Output: [["a"]]

1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.
```

Solution: dict + count

```py
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    # Time: O(N * M), Space: O(N * M)
    d = collections.defaultdict(list)
    for s in strs:
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord("a")] += 1
        d[tuple(counts)].append(s)
    return list(d.values())
```

### Heap

Let's start with max/min heap: **a special binary tree**

The Operations:

- `push`: `O(logn)`, append one element to the tail, then **sift_up** it
- `pop`: `O(logn)`, pop the first element, then move the tail element to first, then **sift_down** (if directly pop, then choosing the child node, the tree structure will be destroyed)
- `top`: `O(1)`

```py
class MinHeap:
    def __init__(self) -> None:
        self.heap = []
    
    def push(self, val: int) -> None:
        self.heap.append(val)
        self._sift_up(len(self.heap) -1)
    
    # sift up the ith element
    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self.heap[index] < self.heap[parent]:
                # changing with parent
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                index = parent
            else:
                break
            
    def pop(self) -> int:
        top = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return top
    
    # sift down the ith element
    def _sift_down(self, index: int) -> None:
        n = len(self.heap)
        # sift down can be adjusted multiple times
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != index:
                # changing with smallest
                self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                index = smallest
            else:
                break
    
    def top(self) -> int:
        return self.heap[0] if self.heap else None
    
    def size(self) -> int:
        return len(self.heap)
```

#### Kth Largest Element in an Array

Q: Given an integer array `nums` and an integer `k`, return the `kth` largest element in the array. Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Eg:

```bash
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

Solution1: Use min heap to maintain the largest k element, so the root is the kth largest number directly. Time: `O(Nlogk)`, Space: `O(N)`

```py
def findKthLargest(nums: List[int], k: int) -> int:
        heap = MinHeap()

        for num in nums:
            if heap.size() < k:
                heap.push(num)
            else:
                if num > heap.top():
                    heap.pop()
                    heap.push(num)
        return heap.top()
```

Solution2: Quick select(part of the quick sort), TODO: [https://leetcode.cn/problems/kth-largest-element-in-an-array/?envType=study-plan-v2&envId=top-interview-150]

## Hard

### Graph

#### Word Ladder

Q: A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

- Every adjacent pair of words differs by a single letter.
- Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
- sk == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

Eg:

```bash
Input: beginWord = "hit", endWord = "cog", wordList = `["hot","dot","dog","lot","log","cog"]`
Output: 5
Explanation: One shortest transformation sequence is `"hit" -> "hot" -> "dot" -> "dog" -> cog"`, which is 5 words long.

Input: beginWord = "hit", endWord = "cog", wordList = `["hot","dot","dog","lot","log"]`
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
```

Solution1: Single BFS, for each index try 26 characters. Time: `O(N*M*26)`, Space: O`(N*M)`

```py
def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])
    visited = set([beginWord])
    word_len = len(beginWord)

    while queue:
        current_word, level = queue.popleft()
        for i in range(word_len):
            # try 26 characters
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == current_word[i]:
                    continue
                next_word = current_word[:i] + c + current_word[i+1:]

                if next_word == endWord:
                    # find end word
                    return level + 1
                
                if next_word in word_set and next_word not in visited:
                    queue.append((next_word, level + 1))
                    visited.add(next_word)
    return 0
```

Solution2: Double BFS. Start from beginning and the end. Time: `O(N*M*26)`, Space: `O(N*M)`.

This could be faster than BFS, eg: if each word can be transformed to 10 words, the path is 6.

- For BFS: `10^6`
- For Double BFS: `10^3 * 2`

```py
def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    begin_set = set([beginWord])
    end_set = set([endWord])
    visited = set([beginWord, endWord])
    word_len = len(beginWord)
    level = 1

    while begin_set and end_set:
        # always start from the smaller set
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set

        # we can't change iterable during iteration, so use a temp to track
        temp = set()
        for word in begin_set:
            for i in range(word_len):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == word[i]:
                        continue
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in end_set:
                        return level + 1
                    if next_word in word_set and next_word not in visited:
                        temp.add(next_word)
                        visited.add(next_word)
        begin_set = temp
        level += 1

    return 0
```

### String

#### Substring with Concatenation of All Words

Q: You are given a string `s` and an array of strings `words`. All the strings of `words` are of **the same length.** A **concatenated string** is a string that exactly contains all the strings of any permutation of `words` concatenated.

Return an array of the starting indices of all the concatenated substrings in `s`. You can return the answer **in any order**.

Eg:

```bash
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation:
The substring starting at 0 is "barfoo". It is the concatenation of ["bar","foo"] which is a permutation of words.
The substring starting at 9 is "foobar". It is the concatenation of ["foo","bar"] which is a permutation of words.

Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []

Input: s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
Output: [6,9,12]
```

Solution: Sliding Window. Note: 1. You should start with each index **in a word length**("abc" eg, you should try `0, 1, 2`). 2. Since **the same length**, we do not need to consider the problem of prefix("a", "ab"). 3. Use two dic to help instead of one.

```py
def findSubstring(s: str, words: List[str]) -> List[int]:
    # Time: `O(N * Length of word)`, Space: `O(Word number)`
    word_length = len(words[0])
    if len(s) < word_length * len(words):
        return []
    word_count = defaultdict(int)
    for word in words:
        word_count[word] += 1
    results = []
    for i in range(word_length):
        left, right, count = i, i, 0
        current_count = defaultdict(int)
        while right + word_length <= len(s):
            word = s[right:right + word_length]
            right += word_length
            if word in word_count:
                current_count[word] += 1
                count += 1
                while current_count[word] > word_count[word]:
                    # word counts doesn't satisfy, move left forward
                    left_word = s[left:left + word_length]
                    current_count[left_word] -= 1
                    count -= 1
                    left += word_length
                if count == len(words):
                    results.append(left)
            else:
                current_count.clear()
                count = 0
                left = right
    return results
```

### Heap

#### IPO

Q: You are given `n` projects where the `ith` project has a pure profit `profits[i]` and a minimum capital of `capital[i]` is needed to start it.

Initially, you have `w` capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.

Pick a list of **at most** `k` distinct projects from given projects to **maximize your final capital**, and return the final maximized capital.

Eg:

```bash
Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
Output: 4
Explanation: Since your initial capital is 0, you can only start the project indexed 0.
After finishing it you will obtain profit 1 and your capital becomes 1.
With capital 1, you can either start the project indexed 1 or the project indexed 2.
Since you can choose at most 2 projects, you need to finish the project indexed 2 to get the maximum capital.
Therefore, output the final maximized capital, which is 0 + 1 + 3 = 4.

Input: k = 3, w = 0, profits = [1,2,3], capital = [0,1,2]
Output: 6
```

Solution: max heap + greedy.

```py
def findMaximizedCapital(k: int, w: int, profits: List, capital: List) -> int:
    # max heap + greedy. Time: O(NlogN), Space: O(N)

    if w >= max(capital):
        # short circuit for special cases
        return w + sum(heapq.nlargest(k, profits))

    l = len(capital)
    cp_packs = [(capital[i], profits[i]) for i in range(l)]
    cp_packs.sort(key=lambda x:x[0])

    heap = []
    i = 0
    for _ in range(k):
        while i < l and cp_packs[i][0] <= w:
            # python uses min heap, so insert with negative profit
            heapq.heappush(heap, -cp_packs[i][1])
            i += 1
        if heap:
            p = heapq.heappop(heap)
            w += -p

    return w
```

Notes:

- For python, default heap is min heap
- Remember to use short circuit for special cases
- `.sort()` should be better than `sorted()`

### Array

#### Candy

Q: There are `n` children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

You are giving candies to these children subjected to the following requirements:

- Each child must have at least one candy.
- Children with a higher rating get more candies than their neighbors.

Return the minimum number of candies you need to have to distribute the candies to the children

Eg:

```bash
Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
Input: ratings = [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
The third child gets 1 candy because it satisfies the above two conditions.
```

Solution 1: Scan twice, Time: `O(N)`, Space: `O(N)`

```py
def candy(ratings: List[int]) -> int:
    n = len(ratings)
    if n == 0:
        return 0

    candies = [1] * n

    # Step 1: Scan from left to right
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Step 2: Scan from right to left
    # Update if the ith rating is higher than the next one
    # Note: Now that the ith candies are more than i-1th, adding more for ith can still satisfy the rule set in step 1
    total_candies = candies[-1]
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
        total_candies += candies[i]

    return total_candies
```

Solution 2: Scan once, Time: `O(N)`, Space: `O(1)`

```python
def candy(ratings: List[int]) -> int:
    result = 1
    increase_len, decrease_len, pre_candies = 1, 0, 1

    for i in range(1, len(ratings)):
        # increasing, adding 1 or pre_candies + 1
        if ratings[i] >= ratings[i - 1]:
            decrease_len = 0
            candies = 1 if ratings[i] == ratings[i - 1] else pre_candies + 1
            result += candies
            increase_len = candies
            pre_candies = candies
        # decreasing
        else:
            decrease_len += 1
            # if the length is the same, eg: 10, 20, 30, 9, 8, 7
            # the candies are: 1, 2, 3, 3, 2, 1
            # so we should add 1 to the length(including the last one of increase)
            # and make the candies like: 1, 2, 4, 3, 2, 1
            if decrease_len == increase_len:
                decrease_len += 1
            
            # adding the length of decrease sequence
            result += decrease_len
            pre_candies = 1     # reset to 1

    return result
```

#### Trap Water

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

Example:

```bash
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Input: height = [4,2,0,3,2,5]
Output: 9
```

My Solution: monotonic stack, Time: `O(N)`, Space: `O(N)`

```py
class Solution:
    def trap(self, height: List[int]) -> int:
        class Block:
            def __init__(self, i: int, value: int) -> None:
                self.i = i
                self.value = value

        s: list[Block] = []
        result = 0
        for i, h in enumerate(height):
            while s and h > s[-1].value:
                v = s.pop()
                if not s:
                    break
                distance = i - s[-1].i - 1
                diff = min(h, s[-1].value) - v.value
                result += diff * distance
            s.append(Block(i, h))
        return result
```

Solution2(skip): Double Pointer, Time: `O(N)`, Space: `O(1)`

```py
def trap(height: List[int]) -> int:
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    result = 0
    
    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            result += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            result += right_max - height[right]
    
    return result
```

Solution3(skip): Dynamic programming. Time: `O(N)`, Space: `O(N)`

```py
def trap(height: List[int]) -> int:
    if not height:
        return 0

    n = len(height)
    left_max = [0] * n
    right_max = [0] * n

    # Fill left_max array
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # Fill right_max array
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # Regions that satisfy both of the max array can trap the rain
    result = 0
    for i in range(n):
        result += min(left_max[i], right_max[i]) - height[i]
    
    return result
```
