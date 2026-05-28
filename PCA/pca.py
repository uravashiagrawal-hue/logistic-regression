import numpy as np
import pandas as pd

# creating dataframe
np.random.seed(23)
mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,o],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1,20)


