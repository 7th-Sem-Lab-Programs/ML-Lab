from sklearn.datasets import load_breast_cancer
from id3 import Id3Estimator
from id3 import export_graphviz

bunch = load_breast_cancer()
estimator = Id3Estimator()
estimator.fit(bunch.data, bunch.target)
export_graphviz(estimator.tree_, 'tree.dot', bunch.feature_names)

