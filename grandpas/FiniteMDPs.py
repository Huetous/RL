def gen(res, l, r, n):
    if l == r == n:
        print(res)
    else:
        if l < n:
            gen(res + "(", l + 1, r, n)
        if r < l:
            gen(res + ")", l, r + 1, n)


gen("", 0, 0, 3)
