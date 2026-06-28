import graphviz.backend as be
from sklearn.datasets import *
from dtreeviz.trees import *
from sklearn import tree
clas = tree.DecisionTreeClassifier()
iris = load_iris()

x_train = iris.data
y_train = iris.target
clas.fit(x_train, y_train)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plot_tree(clas)
plt.show()


# with dtreeviz
import dtreeviz
viz_model = dtreeviz.model(model = clas,X_train=x_train,y_train=y_train, feature_names = iris.feature_names, class_names = iris.target_names, target_name="Species")
# for horizontal decision tree -> orintation='LR'
# for showing node number ->show_node_labels=True
# for removing graphs -> fancy = False
viz = viz_model.view()
viz.show()

regr = tree.DecisionTreeRegressor(max_depth=1)
boston = fetch_california_housing()

X_train = boston.data
y_train = boston.target
regr.fit(X_train, y_train)

viz_model = dtreeviz.model(regr,
               X_train,
               y_train,
               target_name='price',
               feature_names=boston.feature_names,
              )
viz = viz_model.view()
viz.show()
