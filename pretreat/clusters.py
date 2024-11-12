from copy import deepcopy
import numpy as np
import heapq


#  1. mat: 1是存在不共存，0是每次都共存 -> 类间无所谓，类内每次都同时出现
#  2. mat: 1是共存过，0是没共存过 -> 大类间无所谓，大类内每次只出现一个


def objection_func(div_lst, total_num):
    num = 0
    for ls in div_lst:
        num += len(ls)
    if num != total_num:
        return None
    lens = [len(ls) for ls in div_lst]
    return len(div_lst), max(lens) - min(lens)


def is_insert_ok(idx, to_lst, mat):
    for to in to_lst:
        if mat[idx, to] > 0:
            return False
    return True


def is_feasible(mat, div_lst):
    total_num = mat.shape[0]
    num = 0
    for ls in div_lst:
        num += len(ls)
    if num != total_num:
        return False
    for ls in div_lst:
        mat_ = mat[ls]
        mat_ = mat_[:, ls]
        for i in range(mat_.shape[0]):
            mat_[i, i] = 0
        if np.any(mat_ != 0):
            return False
    return True


class DivideNode:
    def __init__(self, mat, div_lst: list[list]):
        self.div_lst = div_lst   # [[0, 2], [4]]
        self.mat = mat           # mat[i, j] is whether i, j coexist
        self.upper = None
        self.lower = None
        self.total_num = mat.shape[0]
        self.feasible_div_lst = None

    def get_upper(self):
        if self.upper is not None:
            return self.upper
        lst = deepcopy(self.div_lst)
        divided = []
        for ls in self.div_lst:
            divided += ls
        divided = set(divided)
        for i in range(self.total_num):
            if i in divided:
                continue
            for ls in lst:
                if is_insert_ok(i, ls, self.mat):
                    ls.append(i)
                    divided.add(i)
                    break
            else:
                lst.append([i])

        self.feasible_div_lst = lst
        self.upper = objection_func(lst, self.total_num)
        return self.upper

    def get_lower(self):
        if self.lower is not None:
            return self.lower
        lens = [len(ls) for ls in self.div_lst]
        if max(lens) > self.total_num // len(self.div_lst):
            max_len = max(lens)
            min_len = (self.total_num - max_len) // (len(self.div_lst) - 1)
            self.lower = len(self.div_lst), max_len - min_len
        else:
            self.lower = len(self.div_lst), int(self.total_num % len(self.div_lst) > 0)
        return self.lower

    def get_new_idx(self):
        divided = []
        for ls in self.div_lst:
            divided += ls
        divided = set(divided)
        if len(divided) == self.total_num:
            return None
        for i in range(self.total_num):
            if i not in divided:
                return i
        return None

    def __lt__(self, other):
        other_lower = other.get_lower()
        lower = self.get_lower()
        if lower[0] < other_lower[0]:
            return True
        if lower[0] > other_lower[0]:
            return False
        if lower[1] < other_lower[1]:
            return True
        return False

    def __gt__(self, other):
        other_lower = other.get_lower()
        lower = self.get_lower()
        if lower[0] < other_lower[0]:
            return False
        if lower[0] > other_lower[0]:
            return True
        if lower[1] > other_lower[1]:
            return True
        return False

    def __le__(self, other):
        other_lower = other.get_lower()
        lower = self.get_lower()
        if lower[0] < other_lower[0]:
            return True
        if lower[0] > other_lower[0]:
            return False
        if lower[1] <= other_lower[1]:
            return True
        return False

    def __ge__(self, other):
        other_lower = other.get_lower()
        lower = self.get_lower()
        if lower[0] < other_lower[0]:
            return False
        if lower[0] > other_lower[0]:
            return True
        if lower[1] >= other_lower[1]:
            return True
        return False

    def __eq__(self, other):
        other_lower = other.get_lower()
        lower = self.get_lower()
        return lower[0] == other_lower[0] and lower[1] == other_lower[1]

    def __str__(self):
        string = f"{self.div_lst}, ({self.lower}, {self.upper})"
        return string


def divide(mat):

    nodes = [DivideNode(mat, [[0]])]
    upper = nodes[0].get_upper()

    results = []

    while len(nodes) > 0:
        node = heapq.heappop(nodes)
        if node.get_lower() > upper:
            continue
        if node.get_upper() < upper:
            upper = node.get_upper()
        idx = node.get_new_idx()
        if idx is None:
            node.lower = node.get_upper()
            results.append(node)
        else:
            for i, ls in enumerate(node.div_lst):
                if is_insert_ok(idx, ls, mat):
                    div_lst = deepcopy(node.div_lst)
                    div_lst[i].append(idx)
                    heapq.heappush(nodes, DivideNode(mat, div_lst))
            div_lst = deepcopy(node.div_lst)
            div_lst.append([idx])
            heapq.heappush(nodes, DivideNode(mat, div_lst))

    return [res.feasible_div_lst for res in results]


if __name__ == '__main__':
    mat = np.zeros([1000, 1000])

    divide(mat)