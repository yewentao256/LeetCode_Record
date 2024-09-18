# LeetCode2024

- [LeetCode2024](#leetcode2024)
  - [Medium](#medium)
    - [Array](#array)
      - [Longest Consecutive Sequence](#longest-consecutive-sequence)
      - [Points That Intersect With Cars](#points-that-intersect-with-cars)
      - [Maximum Points Inside the Square](#maximum-points-inside-the-square)
    - [DFS](#dfs)
      - [Partition to K Equal Sum Subsets](#partition-to-k-equal-sum-subsets)
  - [Hard](#hard)
    - [Find the Median of the Uniqueness Array](#find-the-median-of-the-uniqueness-array)
    - [Find the Maximum Length of a Good Subsequence II](#find-the-maximum-length-of-a-good-subsequence-ii)

## Medium

### Array

#### Longest Consecutive Sequence

Q: Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.

Eg:

```bash
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is `[1, 2, 3, 4]`. Therefore its length is 4.
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

Solutions: Using a set, only start from the head. Time: `O(N)`, Space: `O(N)`

```py
def longestConsecutive(nums: List[int]) -> int:
    if not nums:
        return 0

    num_set = set(nums)  # O(n) time and space
    max_length = 0

    for num in num_set:
        # Only start counting if 'num' is the start of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            # Increment the sequence
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            # Update max_length if current sequence is longer
            max_length = max(max_length, current_length)

    return max_length
```

#### Points That Intersect With Cars

Q: You are given a **0-indexed** 2D integer array nums representing the coordinates of the cars parking on a number line. For any index `i`, `nums[i] = [starti, endi]` where `starti` is the starting point of the `ith` car and `endi` is the ending point of the ith car.

Return the number of integer points on the line that are covered with **any part** of a car.

Eg:

```bash
Input: nums = [[3,6],[1,5],[4,7]]
Output: 7
Explanation: All the points from 1 to 7 intersect at least one car, therefore the answer would be 7.

Input: nums = [[1,3],[5,8]]
Output: 7
Explanation: Points intersecting at least one car are 1, 2, 3, 5, 6, 7, 8. There are a total of 7 points, therefore the answer would be 7.
```

Constraints: (Choose a optimal method!)
1 <= nums.length <= 10000
1 <= starti <= endi <= 10000

Solution: difference array

```py
def numberOfPoints(nums: List[List[int]]) -> int:
    # Time: O(N + L), Space: O(L)

    # Determine Largest L (endpoint)
    L = max(end for _, end in nums)

    # Initialize the difference array
    diff_arr = [0] * (L + 2)  # +2 to handle end + 1
    
    # Update the difference array
    for start, end in nums:
        diff_arr[start] += 1
        diff_arr[end + 1] -= 1
    # Eg: [[3,6],[1,5],[4,7]]
    # index     i    0    1    2    3    4    5     6     7    8
    # diff_arr[i]    0    1    0    1    1    0    -1    -1    -1

    # Compute the prefix sum and count the covered points
    count = 0
    total_points = 0
    for i in range(1, L + 1):   # ith, starting from 1
        count += diff_arr[i]
        if count > 0:
            total_points += 1
    
    return total_points
```

#### Maximum Points Inside the Square

Q & Eg: [https://leetcode.cn/problems/maximum-points-inside-the-square/]

Solution: Maintain the smallest index for all characters, and a smallest limit number. Time: `O(N)`, Space: `O(1)`

```py
def maxPointsInsideSquare(points: List[List[int]], s: str) -> int:
    min_lst = [inf] * 26
    min_square = inf    # Note: We only need to consider one min_square
    for i in range(len(s)):
        x, y = points[i]
        j = ord(s[i]) - ord('a')
        d = max(abs(x), abs(y))
        if d < min_lst[j]:
            min_square = min(min_square, min_lst[j])
            min_lst[j] = d
        elif d < min_square:
            min_square = d
    return sum(d < min_square for d in min_lst)
```

### DFS

#### Partition to K Equal Sum Subsets

Q: Given an integer array `nums` and an integer `k`, return `true` if it is possible to divide this array into `k` non-empty subsets whose sums are all equal.

Eg:

```bash
Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
Explanation: It is possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.

Input: nums = [1,2,3,4], k = 3
Output: false
```

`1 <= k <= nums.length <= 16`

`1 <= nums[i] <= 104`

The frequency of each element is in the range `[1, 4]`.

Solution: memory + dfs + cutoff, Time: `< O(n^k)`, Space: `O(N)`

```py
def canPartitionKSubsets(nums: List[int], k: int) -> bool:
    total = sum(nums)
    if total % k != 0:
        return False
    
    target = total / k
    nums.sort(reverse=True)
    used = [False] * len(nums)

    def dfs(start_index: int, current_sum: int, current_count: int) -> bool:
        if current_count == k - 1:
            # if satisfies k - 1 subsets, then the final set is OK, too
            return True
        if current_sum == target:
            # find a match
            return dfs(0, 0, current_count + 1)

        for i in range(start_index, len(nums)):
            if used[i]:
                continue
            if current_sum + nums[i] > target:
                continue
            used[i] = True
            if dfs(i+1, current_sum + nums[i], current_count):
                return True
            used[i] = False
            if current_sum == 0:
                # For current number, it can not be put into a empty set, this means
                # it can not be used any more, no suitable subsets for current number
                # Eg: [4, 4, 3, 2, 2] with k = 3, target = 5
                # Since you've found 4 is not suitable for any other value
                # So it is impossible to find a subset then
                return False

        return False
    
    return dfs(0, 0, 0)
```

## Hard

### Find the Median of the Uniqueness Array

Q: You are given an integer array nums. The **uniqueness array** of nums is the sorted array that contains the number of distinct elements of all the subarrays of `nums`. In other words, it is a sorted array consisting of distinct(`nums[i..j]`), for all `0 <= i <= j < nums.length`.

Here, `distinct(nums[i..j])` denotes the number of distinct elements in the subarray that starts at index i and ends at index j.

Return the **median** of the **uniqueness array** of `nums`.

Note that the median of an array is defined as the middle element of the array when it is sorted in non-decreasing order. If there are two choices for a median, the **smaller** of the two values is taken.

Eg:

```bash
Input: nums = [1,2,3]
Output: 1
The uniqueness array of nums is [distinct(nums[0..0]), distinct(nums[1..1]), distinct(nums[2..2]), distinct(nums[0..1]), distinct(nums[1..2]), distinct(nums[0..2])] which is equal to [1, 1, 1, 2, 2, 3]. The uniqueness array has a median of 1. Therefore, the answer is 1.

Input: nums = [3,4,3,4,5]
Output: 2
The uniqueness array of nums is [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]. The uniqueness array has a median of 2. Therefore, the answer is 2.

Input: nums = [4,3,5,4]
Output: 2
The uniqueness array of nums is [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]. The uniqueness array has a median of 2. Therefore, the answer is 2.
```

Solution: binary search + sliding window. Time: `O(N*logN)`, Space: `O(N)`

```py
def medianOfUniquenessArray(nums: List[int]) -> int:
    # Solution: binary search(directly find the median) + sliding window
    # the lowest boundary is 1 (at least) and the highest is len(set(nums))
    def count_subarrays_with_median(median: int) -> int:
        # calculate the count of subarrays with given median(max distinct number)
        count = 0           # count of available subarrays
        left = 0
        freq = defaultdict(int)
        distinct_count = 0  # distinct elements in current window
        
        for right in range(len(nums)):
            if freq[nums[right]] == 0:
                distinct_count += 1
            freq[nums[right]] += 1
            
            # sliding window, moving left forward, making sure current window can
            # satisfy with the given median
            while distinct_count > median:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    distinct_count -= 1
                left += 1
            
            # adding all of the subarrys ending with right
            # eg: [1, 2] 
            # count += 1, nums[0...0]
            # count += 2, nums[0...1] and nums[1...1]
            count += right - left + 1
        
        return count

    n = len(nums)
    low, high = 1, len(set(nums))
    median_index = ((n * (n + 1)) // 2 + 1) // 2    # (total count + 1) // 2
    
    while low < high:
        mid = (low + high) // 2
        # if the number of subarrays is smaller than median index
        # it means the median should be bigger, increase low. decrease high otherwise
        if count_subarrays_with_median(mid) < median_index:
            low = mid + 1
        else:
            high = mid
    
    return low
```

### Find the Maximum Length of a Good Subsequence II

Q: You are given an integer array `nums` and a non-negative integer `k`. A sequence of integers `seq` is called good if there are at most `k` indices `i` in the range `[0, seq.length - 2]` such that `seq[i] != seq[i + 1]`.

In other words, at most `k` numbers that are not equal. ([1,2] only count for once)

Return the maximum possible length of a good **subsequence** of nums.

Eg:

```bash
Input: nums = [1,2,1,1,3], k = 2
Output: 4
Explanation:
The maximum length subsequence is [1,2,1,1].

Input: nums = [1,2,3,4,5,1], k = 0
Output: 2
Explanation:
The maximum length subsequence is [1,1].
```

Solution1: Simple 2-D DP. Time: `O(N^2 * k)`, Space: `O(N*k)`

```py
class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # dp[i][l] means ending with nums[i], allows l elements that are not equal
        # dp[i][0] = 1
        # dp[i][l] = for p in (0, i), 
        # if nums[p] == nums[i]: max(dp[p][l-1] + 1) 
        # if nums[p] != nums[i]: max(dp[p][l] + 1)
        dp = [[-1] * (k + 1) for _ in range(len(nums))]
        max_length = 0
        for i in range(len(nums)):
            dp[i][0] = 1
            for l in range(k + 1):
                for p in range(i):
                    if nums[p] == nums[i]:
                        dp[i][l] = max(dp[i][l], dp[p][l] + 1)
                    elif l > 0:  # Only allow changes if l > 0
                        dp[i][l] = max(dp[i][l], dp[p][l - 1] + 1)
                max_length = max(max_length, dp[i][l])

        return max_length
```

Solution2(Optional): [https://www.bilibili.com/video/BV1Tx4y1b7wk/?vd_source=de2754bd08012f6237bf8272aa55de57]
