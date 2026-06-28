import graphviz.backend as be
from sklearn.datasets import *
from dtreeviz.trees import *
from sklearn import tree

clas = tree.DecisionTreeClassifier()
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

sample = iris.data[np.random.randint(0, len(iris.data)),:]

import dtreeviz
viz_model = dtreeviz.model(model = clas,
               X_train = X_train,
               y_train = y_train,
               feature_names=iris.feature_names,
               class_names=list(iris.target_names))
# for showing prediction path only -> show_just_path = True

viz = viz_model.view(x=sample)
viz.show()

