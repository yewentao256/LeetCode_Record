# 2021LeetCode记录

## 简单

## 中等

### 动态规划

#### 秋叶收藏集：Autumn Leaves Collection

- 题目描述：字符串 leaves 仅包含小写字符 r 和 y， 其中字符 r 表示一片红叶，字符 y 表示一片黄叶。要将收藏集中树叶的排列调整成「红、黄、红」三部分。每部分树叶数量可以不相等，但均需大于等于 1。每次调整操作，可以将一片红叶替换成黄叶或者将一片黄叶替换成红叶。最少需要多少次调整操作才能将秋叶收藏集调整完毕？
- 示例：
输入：leaves = "rrryyyrryyyrr"
输出：2
解释：调整两次，将中间的两片红叶替换成黄叶，得到 "rrryyyyyyyyrr"
- 思路：动态规划
    - 划分三种模式：r（全红），ry（红+黄），ryr（红+黄+红）
    - 如果第i片为r：
        - $f(i).r = f(i-1).r$
        - $f(i).ry = min(f(i-1).r, f(i-1).ry) + 1$（翻转），
        - $f(i).ryr = min(f(i-1).ry, f(i-1).ryr)$
    - 如果第i片为y：
        - $f(i).r = f(i-1).r + 1$ （翻转）
        - $f(i).ry = min(f(i-1).r, f(i-1).ry)$
        - $f(i).ryr = min(f(i-1).ry, f(i-1).ryr) + 1$（翻转）
    - 无需数组缓存，从小到大遍历处理即可，初始化r=1/0，ry和ryr都为1
- 时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def minimumOperations(self, leaves: str) -> int:
    #设计三种模式进行：r模式全红，ry模式红+黄，ryr模式红+黄+红
    # 初始化r为0/1，ry、ryr为无穷大
    r = 1 if leaves[0] == 'y' else 0
    ry, ryr = float('inf'), float('inf')

    # 从1开始，从小到大遍历处理
    for i in range(1, len(leaves)):
        #如果第i片为r，r模式不变，ry模式取r模式最小/ry模式最小加1（本次翻转），ryr模式取ry和ryr模式最小
        if leaves[i] == 'r':
            r, ry, ryr = r, min(r, ry) + 1, min(ry, ryr)
        
        #如果第i片为y，r模式+1（本次翻转），ry模式取r模式/ry模式最小，ryr模式取ry和ryr模式最小+1（本次翻转）
        else:
            r, ry, ryr = r + 1, min(r, ry), min(ry, ryr)+1
    return ryr
```
