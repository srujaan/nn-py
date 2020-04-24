import numpy as np

#inputs = [1.3, 4.3, 8.5]
#weights = [3.2, 4.2, 2.1]
#bias = 3


#output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
#print(output)


inp = [2, 12, 1, 4] #1D array
weit = [
        [3, 2, 8, 1],
        [2, 6, 0, 7],
        ]
# weit is 2D array
bias = 0.3
out = np.dot(weit, inp) + bias
print(type(out))
print(out)