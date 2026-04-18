import numpy as np

a =np.array([1,5,3])
b= np.array([-6,-3,-2])
c= np.array([8,2,6])

cosine_similarty = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("b/w a and b", cosine_similarty)

cosine_similarty = np.dot(a,c) / (np.linalg.norm(a) * np.linalg.norm(c))
print("b/w a and c",cosine_similarty)
