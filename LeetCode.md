# 2020LeetCode记录

[TOC]

## 简单

## 中等

### 链表

#### 奇偶链表：Odd Even Linked List——11/13

- 题目说明：输入一个链表，奇数位和偶数位分离
- 样例：
```c
Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
```
- 思路：创建三个指针，odd指向奇数位链表，even指向偶数位链表，even_head记录偶数位链表开头，一次遍历即可。
- 空间复杂度：O（1）， 时间复杂度：O（N）
- 代码：
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        # 没有或者只有一个元素，直接返回
        if not head or not head.next:
            return head
        # 指针初始化
        odd = head
        even_head = odd.next
        even = even_head

        # 遍历——循环展开，一次odd指针和even指针都向后移动两位
        while even and even.next:   # even and even.next一起作为条件，如果even==none，不会执行even.next
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next

        # 遍历完成，将odd接上even_head
        odd.next = even_head
        return head
```



## 困难


