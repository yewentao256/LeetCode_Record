# LeetCode Top 150 Interview

- [LeetCode Top 150 Interview](#leetcode-top-150-interview)
  - [Easy](#easy)
    - [Double Pointer](#double-pointer)
      - [Merge Sorted Array](#merge-sorted-array)
      - [Remove Element](#remove-element)
    - [Array](#array)
      - [Majority Element](#majority-element)
      - [Best Time to Buy and Sell Stock](#best-time-to-buy-and-sell-stock)
  - [Medium](#medium)
    - [Double Pointer](#double-pointer-1)
      - [Remove Duplicates from Sorted Array II](#remove-duplicates-from-sorted-array-ii)

link: [https://leetcode.cn/studyplan/top-interview-150/]

Only questions that I can't pass are recorded

## Easy

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

Solution: Double pointer, **starts from the tail**

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

Solution: Double pointer, **one from the beginning, one from the end**

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
1 <= n <= 5 * 104
```

Solution: **Vote** (Boyer-Moore)

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
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Input: prices = [7,6,4,3,1]
Output: 0
```

Solution: traverse once -- we can't only find the minimum value to buy, we have to consider the max_profit we now can get.

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

Solution: slow and fast pointer

```python
def removeDuplicates(nums: List[int]) -> int:
    # key: judge by key-2, if nums[i-2] == nums[j], nums[i] must == nums[j]
    # key2: nums[i] shoud be covered each time when i moves
    if len(nums) <= 2:
        return len(nums)
    i, j = 2, 2
    while(j < len(nums)):
        if nums[i - 2] == nums[j]:
            # duplicate: j needs to move and find the suitable element
            j += 1
        else:
            # including i==j (first time to cover) and i < j two cases
            nums[i] = nums[j]
            i += 1
            j += 1
        
    return i
```
