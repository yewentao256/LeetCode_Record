# LeetCode2024

- [LeetCode2024](#leetcode2024)
  - [Medium](#medium)
    - [Array](#array)
      - [Maximum Points Inside the Square](#maximum-points-inside-the-square)
    - [DFS](#dfs)
      - [Partition to K Equal Sum Subsets](#partition-to-k-equal-sum-subsets)

## Medium

### Array

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
