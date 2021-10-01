# 程序员面试金典（第六版）

## 简单

### Is Unique

- 题目描述：Implement an algorithm to determine if a string has all unique characters. What if you cannot use additional data structures? all lower case characters. 判断小写字串中字符是否不会出现第二次

- 示例
Input: s = "leetcode"
Output: false

- 代码：`return len(s) == len(set(s))`是个好主意

- 代码2：不使用其他数据结构，用位运算，26个字母转换为26位bit，时间复杂度：O（N），空间复杂度：O（1）

```py
def isUnique(self, astr: str) -> bool:
    mark = 0
    for char in astr:
        move_bit = ord(char) - ord('a')
        if (mark & (1 << move_bit)) != 0:
            return False
        else:
            mark |= (1 << move_bit)
    return True
```

### Check Permutation

- 题目描述：Given two strings,write a method to decide if one is a permutation of the other. 判断是否是另一个字串的排列

- 示例：
Input: s1 = "abc", s2 = "bca"
Output: true

- 代码：

```py
def CheckPermutation(self, s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False
    dic = {}
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            continue
        if dic.get(s1[i]):
            dic[s1[i]] += 1
        else:
            dic[s1[i]] = 1
        # 注：如果用default dict可以不操作get
        if dic.get(s2[i]):
            dic[s2[i]] -= 1
        else:
            dic[s2[i]] = -1

    return set(dic.values()) == {0}
```

### Compress String

- 题目描述：Implement a method to perform basic string compression using the counts of repeated characters. For example, the string `aabcccccaaa` would become `a2blc5a3`. If the "compressed" string would not become smaller than the original string, your method should return the original string. You can assume the string has only uppercase and lowercase letters (a - z).

- 示例：
Input: "aabcccccaaa"
Output: "a2b1c5a3"
Input: "abbccd"
Output: "abbccd"
Explanation: The compressed string is "a1b2c2d1", which is longer than the original string.

- 代码：时间复杂度：O（N），空间复杂度：O（N）（注：用字符串加的话为O（1），但join更有效率）

```py
def compressString(self, S: str) -> str:
    if not S:
        return S
    result = []
    last_character, count = S[0], 1
    for i in range(1, len(S)):
        if last_character == S[i]:
            count += 1
        else:
            result.append(last_character+str(count))
            last_character = S[i]
            count = 1
    
    result.append(last_character+str(count))
    result = ''.join(result)
    
    return result if len(result) < len(S) else S
```

### String Rotation

- 题目描述：Given two strings, s1 and s2, write code to check if s2 is a rotation of s1 (e.g.,"waterbottle" is a rotation of"erbottlewat").

- 示例
Input: s1 = "aa", s2 = "aba"
Output: False

- 代码1：`return len(s1) == len(s2) and s1 in s2*2`
字串旋转过来的，一定首尾相连，字符串in复杂度为O（M+N）（两字串长度），空间复杂度O（1）

- 代码2：首字母开始切片找，时间复杂度：O（N），最坏情况O（N^2），空间复杂度：O（1）

```py
def isFlipedString(self, s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False
    if not s1 and not s2:
        return True
    
    start = s2[0]
    for i in range(len(s1)):
        if s1[i] == start:
            if s1[i:] + s1[:i] == s2:
                return True
    return False
```
