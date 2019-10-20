import numpy as np

foo = [[1, 2, 3], [2, 3, 4], [5, 6, 6]]
bar = [[1,2,3],[2,3,4],[5,6,6]]
new = [[0,0,0],[0,0,0],[0,0,0]]

for i in range(len(foo)):
    for j in range(len(foo[i])):
        new[i][j] = foo[i][j] * bar[i][j]
print(new)

foo = np.array(foo)
bar = np.array(bar)

print(np.multiply(foo, bar))