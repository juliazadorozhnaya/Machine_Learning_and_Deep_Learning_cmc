from typing import List
import math
from functools import reduce


def prod_non_zero_diag(X: List[List[int]]) -> int:
    return reduce(lambda x, y: x * y, t if len(t := [X[i][i] for i in range(min(len(X), len(X[0]))) if X[i][i] != 0]) else [-1])

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    return sorted(x) == sorted(y)


def max_after_zero(x: List[int]) -> int:
    res = -1
    for i in range(1, len(x)):
        if x[i - 1] == 0 and x[i] > res:
            res = x[i]
    return res


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    num_channels = len(weights)
    h = len(image)
    w = len(image[0])
    res_img = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            for ch in range(num_channels):
                res_img[y][x] += weights[ch] * image[y][x][ch]
    return res_img


def run_length_encoding(x: List[int]) -> (List[int], List[int]):
    elems = []
    counters = []
    if len(x) == 0:
        return elems, counters
    elems.append(x[0])
    counters.append(1)
    for elem in x[1:]:
        if elems[-1] == elem:
            counters[-1] += 1
        else:
            elems.append(elem)
            counters.append(1)
    return elems, counters


def pairwise_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    s = 0
    a = []
    for row_x in X:
        temp = []
        for row_y in Y:
            s = 0
            for i in range(len(row_x)):
                s += (row_x[i] - row_y[i]) ** 2
            temp.append(s ** (1 / 2))
        a.append(temp)
    return a