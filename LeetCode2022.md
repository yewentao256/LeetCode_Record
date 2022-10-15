# LeetCode2022

写在前面：仅记录medium和hard的题目和有意思的easy。

## medium

### 动态规划

#### Is Subsequence 判断子序列进阶

- 题目描述：给定字符串 s 和 t ，判断 s 是否为 t 的子序列。很简单对吧？我们现在有大量字符串s需要判断（数量>10亿），怎么处理？
- 示例：
输入：s = "abc", t = "ahbgdc"
输出：true
输入：s = "axc", t = "ahbgdc"
输出：false
- 思路：对于单个s，可以用双指针（代码略）；对于海量的s，我们需要先处理字符串t，获取一个常数量级的字符索引用于判断，即我们设置一个数组或字典，能够O（1） get到s中某个字符在t中的下一个索引。怎么做？扫描t，记录下该字符起的下一个目标字符的索引位置。为了方便扫描，我们使用动态规划（`O（n）* 26`）进行（备注：如果直接暴力获取的话要O（`n^2 * 26`）
- 代码（处理单个s的动态规划代码）：

```python
def isSubsequence(self, s: str, t: str) -> bool:

    len_t, len_s = len(t), len(s)
    # 字符和数字的映射，a -> 0, b -> 1
    dic = {}
    for j in range(26):
        dic[chr(j+ord('a'))] = j
    
    # 1、设定i、j和数组
    # 0<=i<=len_t, 0<=j<=25
    # 假定dp[i][j]为对于t的位置索引i的字符起，字母j首次出现的位置索引
    dp = [[0 for j in range(26)] for i in range(len_t+1)]

    # 2、设置初始值：
    # dp[i][0~25] = len_t       # 可以理解为正无穷（永远取不到）
    # 为方便理解，给出dp[i-1]（最后一个字符，假设为a）的初始值
    # dp[i-1][0] = i-1   # 字符自己
    # dp[i-1][1~25] = len_t
    for j in range(26):
        dp[len_t][j] = len_t

    # 3、状态转移方程
    # dp[i][j] = i if j == t[i]
    # dp[i][j] = dp[i+1][j]
    for i in range(len_t-1, -1, -1):
        for j in range(26):
            dp[i][j] = i if dic[t[i]] == j else dp[i+1][j]
    
    # 4、解决实际问题
    t_index = 0
    for s_index in range(len(s)):
        t_index = dp[t_index][dic[s[s_index]]]
        if t_index == len_t:
            return False
        t_index += 1
    return True
```
