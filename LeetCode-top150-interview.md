# LeetCode Top 150 Interview

- [LeetCode Top 150 Interview](#leetcode-top-150-interview)
  - [Easy](#easy)
    - [Tree](#tree)
      - [Minimum Absolute Difference in BST](#minimum-absolute-difference-in-bst)
    - [Double Pointer](#double-pointer)
      - [Merge Sorted Array](#merge-sorted-array)
      - [Remove Element](#remove-element)
    - [Array](#array)
      - [Majority Element](#majority-element)
      - [Best Time to Buy and Sell Stock](#best-time-to-buy-and-sell-stock)
  - [Medium](#medium)
    - [Double Pointer](#double-pointer-1)
      - [Remove Duplicates from Sorted Array II](#remove-duplicates-from-sorted-array-ii)
    - [Greedy](#greedy)
      - [Jump Game II](#jump-game-ii)
    - [Array](#array-1)
      - [H-Index](#h-index)
      - [Insert Delete GetRandom O(1)](#insert-delete-getrandom-o1)
    - [Dynamic Programming](#dynamic-programming)
      - [Maximum Subarray](#maximum-subarray)

link: [https://leetcode.cn/studyplan/top-interview-150/]

Only questions that I can't pass are recorded

## Easy

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

### Array

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

### Array

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

### Dynamic Programming

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
