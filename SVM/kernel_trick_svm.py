import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

from sklearn.datasets._samples_generator import make_circles
X,y = make_circles(100, factor=.1, noise=.1)
plt.scatter(X[:,0], X[:,1], c=y, s=4, cmap='bwr')
plt.show()
