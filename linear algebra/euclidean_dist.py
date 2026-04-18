import numpy as np
a= np.array([1,3,5,9,8])
b= np.array([6,9,2,1,4])

diff = a-b
print(diff)

# calculate the euclidean dist b/w a and b (L2 norm of the diff)
dist = np.linalg.norm(diff)
print("euclidean dist b/w a and b", dist)

