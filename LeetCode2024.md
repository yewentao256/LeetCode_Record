# LeetCode2024

- [LeetCode2024](#leetcode2024)
  - [Medium](#medium)
    - [Array](#array)
      - [Maximum Points Inside the Square](#maximum-points-inside-the-square)

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
