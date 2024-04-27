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

题目描述：给定字符串 `s` 和 `t` ，判断 `s` 是否为 `t` 的子序列。如果我们现在有大量字符串s需要判断（数量 > 10 亿），怎么处理呢？

示例：

```bash
输入：s = "abc", t = "ahbgdc"
输出：true
输入：s = "axc", t = "ahbgdc"
输出：false
```

思路：对 t 创建一个前进表，判断下一个字符的所处的索引然后快速转移

```python
def isSubsequence(s: str, t: str) -> bool:
    # 时间复杂度：O（M * 26 + N）
    # 空间复杂度：O（M * 26）
    alphabet_size = 26
    # 初始化前进表，记录了字符串 t 中的任一位置 i 出发，下一个期望字符 c 的位置
    # 例如，advance_next[0][1] = 3 表示下一个字符 "b" 的位置在索引3
    # advance_next[3][2] = 6 表示下一个字符 "c" 的位置在索引6
    # -1 表示该字符在当前位置之后不再出现
    advance_next = [[-1] * alphabet_size for _ in range(len(t) + 1)]
    
    # 倒序遍历填充前进表，每次更新一个next_pos，然后整体赋给advance_next[i]
    next_pos = [-1] * alphabet_size
    for i in range(len(t) - 1, -1, -1):
        next_pos[ord(t[i]) - ord('a')] = i
        for j in range(alphabet_size):
            advance_next[i][j] = next_pos[j]
    
    current_index = 0
    for char in s:
        # 如果当前字符在剩余的 t 中没有出现，返回 False
        char_index = ord(char) - ord('a')
        if advance_next[current_index][char_index] == -1:
            return False
        # 移动到下一个匹配字符的位置
        current_index = advance_next[current_index][char_index] + 1
        if current_index > len(t):
            return False
    return True
```
