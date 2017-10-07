u = [0, 0, 41, 32, 0, 0]
v = [0, 0, 0, 0, 0, 0]

for elem in [e for e in enumerate(zip(u, v)) if e[1][0] > 1 and e[1][1] > 1]:
    print(elem)

common_ancestor, (x, y) = min([e for e in enumerate(zip(u, v)) if e[1][0] > 1 and e[1][1] > 1],
                              key=lambda x: x[1][0] + x[1][1],
                              default=(-1, (0, 0)))
print(f"From comprehension: {common_ancestor}, {x}, {y}")
