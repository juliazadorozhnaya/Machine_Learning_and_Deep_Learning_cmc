def check(s, filename):
    d = {}
    for i in s.split():
        if i.lower() in d:
            d[i.lower()] += 1
        else:
            d[i.lower()] = 1
    with open(filename, 'w') as file:
        for key, value in sorted(d.items()):
            file.write(key + ' ' + str(value) + '\n')