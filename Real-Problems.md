# Real-Problems

## 3Sum

Q: Given an integer array nums, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

Eg:

```bash
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].

Input: nums = [0,1,1]
Output: []
Input: nums = [0,0,0]
Output: [[0,0,0]]
```

Solution: Sort the array, then double pointers. Time: `O(N^2)`, Space: `O(1)`(Python `.sort()` only `O(1)`, `sorted()` is `O(N)`)

```py
def threeSum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n):
        # Note: we've sorted the nums, so this equal to the global skip
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # move left and skip duplicate element
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
```

## Maximum Value at a Given Index in a Bounded Array

Q: You are given three positive integers: n, index, and maxSum. You want to construct an array nums (0-indexed) that satisfies the following conditions:

- nums.length == n
- nums[i] is a positive integer where 0 <= i < n.
- abs(nums[i] - nums[i+1]) <= 1 where 0 <= i < n-1.
- The sum of all the elements of nums does not exceed maxSum.
- nums[index] is maximized.

Return nums[index] of the constructed array.

Eg:

```bash
Input: n = 4, index = 2,  maxSum = 6
Output: 2
Explanation: nums = [1,2,2,1] is one array that satisfies all the conditions.
There are no arrays that satisfy all the conditions and have nums[2] == 3, so 2 is the maximum nums[2].

Input: n = 6, index = 1,  maxSum = 10
Output: 3
```

Solution:

```py
def maxValue(n: int, index: int, maxSum: int) -> int:
    # binary search + math validation
    # Time: O(logN), Space: O(1)
    def calculateSum(x):
        left_count = index
        right_count = n - index - 1
        
        # sum of left
        if x > left_count:
            # x is bigger than count, no need to worry "1"
            sum_left = (x + (x - left_count)) * (left_count + 1) // 2
        else:
            sum_left = (x * (x + 1)) // 2 + (left_count - x + 1)
        
        if x > right_count:
            sum_right = (x + (x - right_count)) * (right_count + 1) // 2
        else:
            sum_right = (x * (x + 1)) // 2 + (right_count - x + 1)
        
        total_sum = sum_left + sum_right - x
        
        return total_sum
    
    # binary search
    low, high = 1, maxSum
    
    while low < high:
        mid = (low + high + 1) // 2
        if calculateSum(mid) <= maxSum:
            low = mid
        else:
            high = mid - 1
    
    return low
```
