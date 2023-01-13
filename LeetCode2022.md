# LeetCode2022

写在前面：仅记录medium和hard的题目和有意思的easy。

## easy

### Rearrange Characters to Make Target String：重排字符形成目标字符串

- 题目描述：给你两个下标从 0 开始的字符串 s 和 target 。你可以从 s 取出一些字符并将其重排，得到若干新的字符串。从 s 中取出字符并重新排列，返回可以形成 target 的 最大 副本数。

- 示例：

输入：s = "ilovecodingonleetcode", target = "code"
输出：2
解释：
对于 "code" 的第 1 个副本，选取下标为 4 、5 、6 和 7 的字符。
对于 "code" 的第 2 个副本，选取下标为 17 、18 、19 和 20 的字符。
形成的字符串分别是 "ecod" 和 "code" ，都可以重排为 "code" 。
可以形成最多 2 个 "code" 的副本，所以返回 2 。

- 思路：哈希表计数，可以用整除的方式一次遍历
- 代码：

```py
from collections import defaultdict
class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        dic = defaultdict(int)
        dic_target = defaultdict(int)
        for c in s:
            dic[c] += 1
        for c in target:
            dic_target[c] += 1
        
        result = inf
        for key, count in dic_target.items():
            # 不需要检测是否key是否存在，defaultdict(int)取不存在的key时返回0
            result = min(result, dic[key] // count)
            if result == 0:
                return 0
        return result
```

### Longest Common Prefix：最长公共前缀

- 题目描述：编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串 ""。
- 示例：
输入：strs = ["flower","flow","flight"]
输出："fl"
- 思路：横向扫描，每次更新最长前缀。时间复杂度O(MN)，M是平均字符串长度，最坏情况下都要比较一次，空间复杂度O（1）
- 代码

```py
def longestCommonPrefix(self, strs: List[str]) -> str:
    prefix = strs[0]

    for i in range(1, len(strs)):
        compare_str = strs[i]
        j, l = 0, min(len(prefix), len(compare_str))
        while j < l:
            if prefix[j] != compare_str[j]:
                break
            j += 1
        prefix = prefix[:j]
    
    return prefix
    
```

### Valid Parentheses：有效的括号

- 题目描述：给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。

- 示例：
输入：s = ""([)]""
输出：false
输入：s = "()[]{}"
输出：true

- 思路：一般这种匹配都是用栈，先进后出一一对应，注意边界条件如栈空的情况

- 代码：

```py
class Solution:
    def isValid(self, s: str) -> bool:
        stack = ['bottom']
        dic = {'{': '}', '[': ']', '(': ')', 'bottom': 'bottom'}

        for c in s:
            if dic.get(c):
                stack.append(c)
            else:
                if dic[stack.pop()] != c:
                    return False
        return len(stack) == 1

```

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
