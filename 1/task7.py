def process(l):
    return sorted(i**2 for i in set(sum(l, [])))[::-1]