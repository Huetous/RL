import operator


class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.operation = operation
        self.tree = [neutral_element for _ in range(2 * capacity)]
        self.neutral_element = neutral_element

    def reduce(self, start=0, end=None):
        assert 0 <= start, end < self.capacity
        if end is None:
            end = self.capacity - 1

        start += self.capacity
        end += self.capacity
        stat = self.neutral_element

        while start <= end:
            if start % 2 != 0:
                stat = self.operation(
                    self.tree[start],
                    stat
                )

            if end % 2 == 0:
                stat = self.operation(
                    self.tree[end],
                    stat
                )

            start = (start + 1) // 2
            end = (end - 1) // 2
        return stat

    def __setitem__(self, index, value):
        assert 0 <= index < self.capacity
        index += self.capacity
        self.tree[index] = value

        index //= 2
        while index >= 1:
            self.tree[index] = self.operation(
                self.tree[2 * index],
                self.tree[2 * index + 1]
            )
            index //= 2

    def __getitem__(self, index):
        assert 0 <= index < self.capacity
        return self.tree[index + self.capacity]


class SumTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super().reduce(start, end)

    def find_prefix_sum(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        index = 1
        while index < self.capacity:
            if self.tree[2 * index] > prefixsum:
                index = 2 * index
            else:
                prefixsum -= self.tree[2 * index]
                index = 2 * index + 1
        return index - self.capacity


class MinTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float("inf")
        )

    def min(self, start=0, end=None):
        return super().reduce(start, end)


if __name__ == "__main__":
    n = 10

    capacity = 1
    while capacity < n:
        capacity *= 2

    st = SumTree(capacity)
    mt = MinTree(capacity)
    for i in range(n):
        st[i] = 1 / (i + 1)
        mt[i] = 1 / (i + 1)

    total = st.sum()

    l = total / 5
    prefixsum = 2*l + l*0.7
    print(st.find_prefix_sum(prefixsum))

    assert st.sum() == n * (n + 1) / 2
    assert st.sum(6, 6) == 7
    assert st.sum(98, 99) == 199

    assert mt.min() == 1
    assert mt.min(6, 6) == 7
    assert mt.min(98, 99) == 99
