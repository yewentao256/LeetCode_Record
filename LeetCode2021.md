# 2021LeetCode记录

## 简单

### 动态规划

#### 斐波那契数列：Fibonacci Number

- 问题描述：输入n，返回对应斐波那契数列值
- 示例：输入2，返回1；输入3，返回2
- 思路：动态规划+不缓存减少空间复杂度。时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def fib(self, n: int) -> int:
    if n <2:
        return n
    p, q, r = 0,0,1
    for i in range(2,n+1):
        p,q = q,r
        r = p+q
    return r 
```
- 注：如果用通项公式可以O（1）的时间复杂度

### 贪心

#### 种花问题：Can Place Flowers

- 问题描述：长花坛的花不能种植在相邻的地块上。一个整数数组flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种花，1 表示种植了花。另有一个数 n ，能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false。
- 样例
输入：flowerbed = [1,0,0,0,1], n = 1
输出：true
- 思路：贪心遍历即可，注意边界值检查，为此我们引入哨兵。时间复杂度：O（N），空间复杂度：O（1）
- 代码：
```python
def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    # 尾部添加哨兵值
    flowerbed = flowerbed + [0]
    i,length = 0,len(flowerbed)-1

    while i < length:
        # 如果当前位置有花，则当前位置和下一位置都不能种花
        if flowerbed[i]:
            i += 2
        # 如果下一位置有花，那么需要直接i+3，到下下下位置
        elif flowerbed[i+1]:
            i += 3
        # 当前位置和下一位置都无花，在此处种下，只需要修改n不需要修改数组
        else:
            n -=1
            i += 2
    return n<=0
```

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
