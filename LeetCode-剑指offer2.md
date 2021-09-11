# 剑指offer2

## 简单

### 两个栈实现队列

- 题目说明：用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

- 代码：

```py
class CQueue:

    def __init__(self):
        self.put_stack = []
        self.get_stack = []

    def appendTail(self, value: int) -> None:
        self.put_stack.append(value)

    def deleteHead(self) -> int:
        if self.get_stack:
            return self.get_stack.pop()
        while self.put_stack:
            self.get_stack.append(self.put_stack.pop())
        # 如果有数据，return 数据，否则return -1
        if self.get_stack:
            return self.get_stack.pop()
        return -1
```

- 时间复杂度：O（1），因为弹出操作对每个元素最多只操作一次，因此均摊下来为O（1）；空间复杂度：O（N）

### 斐波那契数列

- 题目说明：略

- 代码：

```py
def fib(self, n: int) -> int:
    MOD = 10 ** 9 + 7
    if n < 2:
        return n
    p, q, r = 0, 0, 1
    for i in range(2, n + 1):
        p = q
        q = r
        r = p + q
    return r % MOD
```

- 时间复杂度：O（N），空间复杂度：O（1）
- 注：如果用矩阵快速幂的方法可以将时间复杂度降到O（logn）
- 注：先算好再给的方法，初始化O（N）之后，每次fib都是O（1）的时间复杂度
