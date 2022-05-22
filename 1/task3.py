def longestCommonPrefix(x):

    if len(x) == 1:
        return x[0]

    x = [i.strip(' ') for i in x]
    prefix = x[0]

    for string in x[1:]:
        while string[:len(prefix)] != prefix and prefix:
            prefix = prefix[:len(prefix)-1]
        if not prefix or prefix == " ":
            return ""
            break

    return prefix
