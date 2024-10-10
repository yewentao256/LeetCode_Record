# LeetCode2024

- [LeetCode2024](#leetcode2024)
  - [Easy](#easy)
    - [Array](#array)
      - [Sort Array by Increasing Frequency](#sort-array-by-increasing-frequency)
  - [Medium](#medium)
    - [Binary search](#binary-search)
      - [Maximum Candies Allocated to K Children](#maximum-candies-allocated-to-k-children)
    - [Math](#math)
      - [Airplane Seat Assignment Probability](#airplane-seat-assignment-probability)
    - [Heap](#heap)
      - [Seat Reservation Manager](#seat-reservation-manager)
    - [Sliding Window](#sliding-window)
      - [Take K of Each Character From Left and Right](#take-k-of-each-character-from-left-and-right)
    - [Array](#array-1)
      - [Longest Consecutive Sequence](#longest-consecutive-sequence)
      - [Points That Intersect With Cars](#points-that-intersect-with-cars)
      - [Maximum Points Inside the Square](#maximum-points-inside-the-square)
    - [DFS](#dfs)
      - [Partition to K Equal Sum Subsets](#partition-to-k-equal-sum-subsets)
  - [Hard](#hard)
    - [Find Subarray With Bitwise OR Closest to K](#find-subarray-with-bitwise-or-closest-to-k)
    - [Minimum Cost to Reach Destination in Time(Example)](#minimum-cost-to-reach-destination-in-timeexample)
    - [Maximum Number of Robots Within Budget](#maximum-number-of-robots-within-budget)
    - [Find the Median of the Uniqueness Array](#find-the-median-of-the-uniqueness-array)
    - [Find the Maximum Length of a Good Subsequence II](#find-the-maximum-length-of-a-good-subsequence-ii)

## Easy

### Array

#### Sort Array by Increasing Frequency

Q: Given an array of integers nums, sort the array in increasing order based on the frequency of the values. If multiple values have the same frequency, sort them in decreasing order.

Return the sorted array.

Eg:

```bash
Input: nums = [1,1,2,2,2,3]
Output: [3,1,1,2,2,2]
Explanation: '3' has a frequency of 1, '1' has a frequency of 2, and '2' has a frequency of 3.

Input: nums = [2,3,1,3,2]
Output: [1,3,3,2,2]
Explanation: '2' and '3' both have a frequency of 2, so they are sorted in decreasing order.

Input: nums = [-1,1,-6,4,5,-6,1,4,1]
Output: [5,-1,4,4,-6,-6,1,1,1]

```

Solution: Use `.sort()`

```py
def frequencySort(nums: List[int]) -> List[int]:
    # Time: O(NlogN), Space: O(N)
    from collections import Counter

    counter = Counter(nums)
    # (counter[x], -x) means
    # 1. choose frequency as the primary element
    # 2. choose -x as the second element(to descend)
    nums.sort(key=lambda x: (counter[x], -x))
    return nums
```

## Medium

### Binary search

#### Maximum Candies Allocated to K Children

Q: You are given a 0-indexed integer array candies. Each element in the array denotes a pile of candies of size candies[i]. You can divide each pile into any number of sub piles, but you cannot merge two piles together.

You are also given an integer k. You should allocate piles of candies to k children such that each child gets the same number of candies. Each child can take at most one pile of candies and some piles of candies may go unused.

Return the maximum number of candies each child can get.

Eg:

```bash
Input: candies = [5,8,6], k = 3
Output: 5
Explanation: We can divide candies[1] into 2 piles of size 5 and 3, and candies[2] into 2 piles of size 5 and 1. We now have five piles of candies of sizes 5, 5, 3, 5, and 1. We can allocate the 3 piles of size 5 to 3 children. It can be proven that each child cannot receive more than 5 candies.

Input: candies = [2,5], k = 11
Output: 0
Explanation: There are 11 children but only 7 candies in total, so it is impossible to ensure each child receives at least one candy. Thus, each child gets no candy and the answer is 0.
```

Solution: Binary search. Time: `O(N logM)`, Space: `O(1)`

```py
def maximumCandies(candies: List[int], k: int) -> int:
    # Check if the given amount is enough for k children
    def check(amount: int) -> bool:
        s = 0
        for candy in candies:
            s += candy // amount
        return s >= k

    # Define binary search boundaries, right should be maximum + 1
    left, right = 1, max(candies) + 1
    
    # Perform binary search to find the minimum amount that is not satisfied
    # So the maximum amount that is satisfied is left - 1
    # Note: after break, left == right
    while left < right:
        mid = (left + right) // 2
        if check(mid):
            left = mid + 1
        else:
            right = mid
    
    return left - 1
```

### Math

#### Airplane Seat Assignment Probability

Q: n passengers board an airplane with exactly n seats. The first passenger has lost the ticket and picks a seat randomly. But after that, the rest of the passengers will:

- Take their own seat if it is still available, and
- Pick other seats randomly when they find their seat occupied

Return the probability that the nth person gets his own seat.

Eg:

```bash
Example 1:

Input: n = 1
Output: 1.00000
Explanation: The first person can only get the first seat.
Example 2:

Input: n = 2
Output: 0.50000
Explanation: The second person has a probability of 0.5 to get the second seat (when first person gets the first seat).
```

Solution: Mathematical induction. Time: `O(1)`, Space: `O(1)`

```py
def nthPersonGetsNthSeat(n: int) -> float:
    # f(1) = 1, f(2) = 0.5, calculate and get f(3) = 0.5, f(4) = 0.5
    # Mathematical induction: assume f(k) = 0.5
    # f(k + 1) = 1 / (k+1) + 1/ (k+1) * f(k) * (k-1) + 0 = 0.5
    # Prove done
    if n == 1:
        return 1
    else:
        return 0.5
```

### Heap

#### Seat Reservation Manager

Q: Design a system that manages the reservation state of n seats that are numbered from 1 to n.

Implement the SeatManager class:

- `SeatManager(int n)` Initializes a SeatManager object that will manage n seats numbered from 1 to n. All seats are initially available.
- `int reserve()` Fetches the smallest-numbered unreserved seat, reserves it, and returns its number.
- `void unreserve(int seatNumber)` Unreserves the seat with the given seatNumber.

Note: For each call to reserve, it is guaranteed that there will be at least one unreserved seat. For each call to unreserve, it is guaranteed that seatNumber will be reserved.

Eg:

```bash
Input
["SeatManager", "reserve", "reserve", "unreserve", "reserve", "reserve", "reserve", "reserve", "unreserve"]
[[5], [], [], [2], [], [], [], [], [5]]
Output
[null, 1, 2, null, 2, 3, 4, 5, null]

Explanation
SeatManager seatManager = new SeatManager(5); // Initializes a SeatManager with 5 seats.
seatManager.reserve();    // All seats are available, so return the lowest numbered seat, which is 1.
seatManager.reserve();    // The available seats are [2,3,4,5], so return the lowest of them, which is 2.
seatManager.unreserve(2); // Unreserve seat 2, so now the available seats are [2,3,4,5].
seatManager.reserve();    // The available seats are [2,3,4,5], so return the lowest of them, which is 2.
seatManager.reserve();    // The available seats are [3,4,5], so return the lowest of them, which is 3.
seatManager.reserve();    // The available seats are [4,5], so return the lowest of them, which is 4.
seatManager.reserve();    // The only available seat is seat 5, so return 5.
seatManager.unreserve(5); // Unreserve seat 5, so now the available seats are [5].
```

Solution: Min heap, Time: `O(LogN)`, Space: `O(N)`

```py
import heapq
class SeatManager:
    def __init__(self, n: int):
        # No need to initialize the heap with all seats to keep it efficient.
        # We'll dynamically add unreserved seats to the heap.
        self.available_seats = []
        self.next_seat = 0

    def reserve(self) -> int:
        if self.available_seats:
            return heapq.heappop(self.available_seats)
        else:
            self.next_seat += 1
            return self.next_seat

    def unreserve(self, seatNumber: int) -> None:
        # Add the seat back to the heap if it's within the range and not already unreserved.
        # Since the problem guarantees that unreserve is called only on reserved seats,
        # we don't need to check for duplicates.
        heapq.heappush(self.available_seats, seatNumber)
```

### Sliding Window

#### Take K of Each Character From Left and Right

Q: You are given a string s consisting of the characters 'a', 'b', and 'c' and a non-negative integer k. Each minute, you may take either the leftmost character of s, or the rightmost character of s.

Return the **minimum** number of minutes needed for you to take **at least** k of each character, or return `-1` if it is not possible to take k of each character.

Eg:

```bash
Input: s = "aabaaaacaabc", k = 2
Output: 8
Explanation: 
Take three characters from the left of s. You now have two 'a' characters, and one 'b' character.
Take five characters from the right of s. You now have four 'a' characters, two 'b' characters, and two 'c' characters.
A total of 3 + 5 = 8 minutes is needed.
It can be proven that 8 is the minimum number of minutes needed.

Input: s = "a", k = 1
Output: -1
Explanation: It is not possible to take one 'b' or 'c' so return -1.
```

Solution:

```py
def takeCharacters(s: str, k: int) -> int:
    # Solution: Sliding window
    # Time: O(N), Space: O(1)
    # Reverse thinking: get the maximum substring that satisfy a b c all >= k outside
    # So we firstly count all of the characters, as window grows, we subtract the count
    counts = [0] * 3
    min_ops = len(s)

    for c in s:
        counts[ord(c) - ord('a')] += 1
    
    # return earlier if not satisfied
    if not (counts[0] >= k and counts[1] >= k and counts[2] >= k):
        return -1

    # sliding window
    left = 0
    for right, c in enumerate(s):
        counts[ord(c) - ord('a')] -= 1
        while left < right and (counts[0] < k or counts[1] < k or counts[2] < k):
            counts[ord(s[left]) - ord('a')] += 1
            left += 1
        if counts[0] >= k and counts[1] >= k and counts[2] >= k:
            min_ops = min(min_ops, len(s) - (right - left + 1))

    return min_ops
```

### Array

#### Longest Consecutive Sequence

Q: Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.

Eg:

```bash
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is `[1, 2, 3, 4]`. Therefore its length is 4.
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
```

Solutions: Using a set, only start from the head. Time: `O(N)`, Space: `O(N)`

```py
def longestConsecutive(nums: List[int]) -> int:
    if not nums:
        return 0

    num_set = set(nums)  # O(n) time and space
    max_length = 0

    for num in num_set:
        # Only start counting if 'num' is the start of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            # Increment the sequence
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            # Update max_length if current sequence is longer
            max_length = max(max_length, current_length)

    return max_length
```

#### Points That Intersect With Cars

Q: You are given a **0-indexed** 2D integer array nums representing the coordinates of the cars parking on a number line. For any index `i`, `nums[i] = [starti, endi]` where `starti` is the starting point of the `ith` car and `endi` is the ending point of the ith car.

Return the number of integer points on the line that are covered with **any part** of a car.

Eg:

```bash
Input: nums = [[3,6],[1,5],[4,7]]
Output: 7
Explanation: All the points from 1 to 7 intersect at least one car, therefore the answer would be 7.

Input: nums = [[1,3],[5,8]]
Output: 7
Explanation: Points intersecting at least one car are 1, 2, 3, 5, 6, 7, 8. There are a total of 7 points, therefore the answer would be 7.
```

Constraints: (Choose a optimal method!)
1 <= nums.length <= 10000
1 <= starti <= endi <= 10000

Solution: difference array

```py
def numberOfPoints(nums: List[List[int]]) -> int:
    # Time: O(N + L), Space: O(L)

    # Determine Largest L (endpoint)
    L = max(end for _, end in nums)

    # Initialize the difference array
    diff_arr = [0] * (L + 2)  # +2 to handle end + 1
    
    # Update the difference array
    for start, end in nums:
        diff_arr[start] += 1
        diff_arr[end + 1] -= 1
    # Eg: [[3,6],[1,5],[4,7]]
    # index     i    0    1    2    3    4    5     6     7    8
    # diff_arr[i]    0    1    0    1    1    0    -1    -1    -1

    # Compute the prefix sum and count the covered points
    count = 0
    total_points = 0
    for i in range(1, L + 1):   # ith, starting from 1
        count += diff_arr[i]
        if count > 0:
            total_points += 1
    
    return total_points
```

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

## Hard

### Find Subarray With Bitwise OR Closest to K

Q: You are given an array nums and an integer k. You need to find a subarray of nums such that the absolute difference between k and the bitwise OR of the subarray elements is as small as possible. In other words, select a subarray nums[l..r] such that |k - (nums[l] OR nums[l + 1] ... OR nums[r])| is minimum.

Return the minimum possible value of the absolute difference.

Eg:

```bash
Input: nums = [1,2,4,5], k = 3
Output: 0
The subarray nums[0..1] has OR value 3, which gives the minimum absolute difference |3 - 3| = 0.

Input: nums = [1,3,1,3], k = 2
Output: 1
The subarray nums[1..1] has OR value 3, which gives the minimum absolute difference |3 - 2| = 1.

Input: nums = [1], k = 10
Output: 9
There is a single subarray with OR value 1, which gives the minimum absolute difference |10 - 1| = 9.
```

Solution: Set + `Or` feature

```py
def minimumDifference(nums: List[int], k: int) -> int:
    # Note: Since or can only have 32 kinds, so we use set
    # Time: O(N * bits), Space: O(bits)

    prev_ors = set()
    min_diff = float('inf')
    
    for num in nums:
        current_ors = set()
        # current_ors init with num it self
        current_ors.add(num)
        
        # here we get the result from all before
        # eg: [0~n], [1~n], [2~n]... [n-1, n]
        for or_val in prev_ors:
            current_ors.add(or_val | num)
        
        # update min_diff
        for or_val in current_ors:
            diff = abs(k - or_val)
            if diff < min_diff:
                min_diff = diff
                if min_diff == 0:
                    return 0
        
        prev_ors = current_ors
    
    return min_diff
```

### Minimum Cost to Reach Destination in Time(Example)

Q and Eg: [Leetcode](https://leetcode.cn/problems/minimum-cost-to-reach-destination-in-time/description/?envType=daily-question&envId=2024-10-03)

Solution: Dijkstra + Heap. Time: `O(E + E * logE)`, Space: `O(E + N)`

Note: `logE` for handling the heap.

```py
def minCost(maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
    # Build the graph as an adjacency list
    graph = defaultdict(list)
    for x, y, time in edges:
        graph[x].append((y, time))
        graph[y].append((x, time))
    
    # Min heap: (current_cost, current_node, current_time). Start with city 0
    heap = [(passingFees[0], 0, 0)]
    
    # Initialize min_time array to keep track of the minimum time to reach each city
    min_time = [float('inf')] * len(passingFees)
    min_time[0] = 0
    
    while heap:
        current_cost, current_node, current_time = heapq.heappop(heap)
        
        # If we've reached the destination within the allowed time, return the cost
        if current_node == len(passingFees) - 1 and current_time <= maxTime:
            return current_cost
        
        # Explore neighboring cities
        for neighbor, travel_time in graph[current_node]:
            new_time = current_time + travel_time
            if new_time > maxTime:
                continue

            new_cost = current_cost + passingFees[neighbor]
            # If we've found a faster way to reach the neighbor, consider this path
            # Note: Will it overwrite the smallest cost? No
            # Because we always choose the smallest cost to go, if we arrive within the time, we return earlier.
            if new_time < min_time[neighbor]:
                min_time[neighbor] = new_time
                heapq.heappush(heap, (new_cost, neighbor, new_time))
    # If destination is not reachable within maxTime
    return -1

```

### Maximum Number of Robots Within Budget

Q: You have n robots. You are given two 0-indexed integer arrays, chargeTimes and runningCosts, both of length n. The ith robot costs chargeTimes[i] units to charge and costs runningCosts[i] units to run. You are also given an integer budget.

The total cost of running k chosen robots is equal to max(chargeTimes) + k * sum(runningCosts), where max(chargeTimes) is the largest charge cost among the k robots and sum(runningCosts) is the sum of running costs among the k robots.

Return the maximum number of consecutive robots you can run such that the total cost does not exceed budget.

Eg:

```bash
Input: chargeTimes = [3,6,1,3,4], runningCosts = [2,1,3,4,5], budget = 25
Output: 3
Explanation: 
It is possible to run all individual and consecutive pairs of robots within budget.
To obtain answer 3, consider the first 3 robots. The total cost will be max(3,6,1) + 3 * sum(2,1,3) = 6 + 3 * 6 = 24 which is less than 25.
It can be shown that it is not possible to run more than 3 consecutive robots within budget, so we return 3.

Input: chargeTimes = [11,12,19], runningCosts = [10,8,7], budget = 19
Output: 0
Explanation: No robot can be run that does not exceed the budget, so we return 0.
```

Solution: Sliding window + monotonic queue

```py
def maximumRobots(chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
    # Time: O(N), Space: O(N)
    left = 0
    running_sum = 0
    max_charge = 0
    max_queue = deque()
    max_length = 0

    for right in range(len(chargeTimes)):
        running_sum += runningCosts[right]

        # maintain the monotonic queue, keeping the head maximum chargetime
        while max_queue and chargeTimes[right] > chargeTimes[max_queue[-1]]:
            max_queue.pop()
        max_queue.append(right)
        max_charge = chargeTimes[max_queue[0]]
        
        # Calculating the total cost
        window_size = right - left + 1
        current_cost = max_charge + window_size * running_sum
        
        # Out of budget, moving left
        while current_cost > budget and left <= right:
            running_sum -= runningCosts[left]
            if max_queue[0] == left:
                max_queue.popleft()
            left += 1
            if max_queue:
                max_charge = chargeTimes[max_queue[0]]
            else:
                max_charge = 0
            window_size = right - left + 1
            current_cost = max_charge + window_size * running_sum
        
        max_length = max(max_length, right - left + 1)        
    return max_length
```

### Find the Median of the Uniqueness Array

Q: You are given an integer array nums. The **uniqueness array** of nums is the sorted array that contains the number of distinct elements of all the subarrays of `nums`. In other words, it is a sorted array consisting of distinct(`nums[i..j]`), for all `0 <= i <= j < nums.length`.

Here, `distinct(nums[i..j])` denotes the number of distinct elements in the subarray that starts at index i and ends at index j.

Return the **median** of the **uniqueness array** of `nums`.

Note that the median of an array is defined as the middle element of the array when it is sorted in non-decreasing order. If there are two choices for a median, the **smaller** of the two values is taken.

Eg:

```bash
Input: nums = [1,2,3]
Output: 1
The uniqueness array of nums is [distinct(nums[0..0]), distinct(nums[1..1]), distinct(nums[2..2]), distinct(nums[0..1]), distinct(nums[1..2]), distinct(nums[0..2])] which is equal to [1, 1, 1, 2, 2, 3]. The uniqueness array has a median of 1. Therefore, the answer is 1.

Input: nums = [3,4,3,4,5]
Output: 2
The uniqueness array of nums is [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]. The uniqueness array has a median of 2. Therefore, the answer is 2.

Input: nums = [4,3,5,4]
Output: 2
The uniqueness array of nums is [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]. The uniqueness array has a median of 2. Therefore, the answer is 2.
```

Solution: binary search + sliding window. Time: `O(N*logN)`, Space: `O(N)`

```py
def medianOfUniquenessArray(nums: List[int]) -> int:
    # Solution: binary search(directly find the median) + sliding window
    # the lowest boundary is 1 (at least) and the highest is len(set(nums))
    def count_subarrays_with_median(median: int) -> int:
        # calculate the count of subarrays with given median(max distinct number)
        count = 0           # count of available subarrays
        left = 0
        freq = defaultdict(int)
        distinct_count = 0  # distinct elements in current window
        
        for right in range(len(nums)):
            if freq[nums[right]] == 0:
                distinct_count += 1
            freq[nums[right]] += 1
            
            # sliding window, moving left forward, making sure current window can
            # satisfy with the given median
            while distinct_count > median:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    distinct_count -= 1
                left += 1
            
            # adding all of the subarrys ending with right
            # eg: [1, 2] 
            # count += 1, nums[0...0]
            # count += 2, nums[0...1] and nums[1...1]
            count += right - left + 1
        
        return count

    n = len(nums)
    low, high = 1, len(set(nums))
    median_index = ((n * (n + 1)) // 2 + 1) // 2    # (total count + 1) // 2
    
    while low < high:
        mid = (low + high) // 2
        # if the number of subarrays is smaller than median index
        # it means the median should be bigger, increase low. decrease high otherwise
        if count_subarrays_with_median(mid) < median_index:
            low = mid + 1
        else:
            high = mid
    
    return low
```

### Find the Maximum Length of a Good Subsequence II

Q: You are given an integer array `nums` and a non-negative integer `k`. A sequence of integers `seq` is called good if there are at most `k` indices `i` in the range `[0, seq.length - 2]` such that `seq[i] != seq[i + 1]`.

In other words, at most `k` numbers that are not equal. ([1,2] only count for once)

Return the maximum possible length of a good **subsequence** of nums.

Eg:

```bash
Input: nums = [1,2,1,1,3], k = 2
Output: 4
Explanation:
The maximum length subsequence is [1,2,1,1].

Input: nums = [1,2,3,4,5,1], k = 0
Output: 2
Explanation:
The maximum length subsequence is [1,1].
```

Solution1: Simple 2-D DP. Time: `O(N^2 * k)`, Space: `O(N*k)`

```py
class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # dp[i][l] means ending with nums[i], allows l elements that are not equal
        # dp[i][0] = 1
        # dp[i][l] = for p in (0, i), 
        # if nums[p] == nums[i]: max(dp[p][l-1] + 1) 
        # if nums[p] != nums[i]: max(dp[p][l] + 1)
        dp = [[-1] * (k + 1) for _ in range(len(nums))]
        max_length = 0
        for i in range(len(nums)):
            dp[i][0] = 1
            for l in range(k + 1):
                for p in range(i):
                    if nums[p] == nums[i]:
                        dp[i][l] = max(dp[i][l], dp[p][l] + 1)
                    elif l > 0:  # Only allow changes if l > 0
                        dp[i][l] = max(dp[i][l], dp[p][l - 1] + 1)
                max_length = max(max_length, dp[i][l])

        return max_length
```

Solution2(Optional): [https://www.bilibili.com/video/BV1Tx4y1b7wk/?vd_source=de2754bd08012f6237bf8272aa55de57]
