import numpy as np
import pandas as df
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import load_wine
from collections import Counter #ALLOWS US TO COUNT OBJECTS IN A LIST, RETURNS A DICT
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #TO PLOT TREE
from sklearn.metrics import accuracy_score, classification_report
from sklearn . datasets import fetch_openml
from MyRandomForestClassifier import MyRandomForestClassifier

dataset = fetch_openml ("mnist_784")
X = dataset["data"]
y = dataset["target"].astype(int)
feature_names = dataset["feature_names"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=10000, random_state=42, shuffle=True
)

print(X.shape)
print(y.shape)
#print(feature_names)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#print(accuracy_score(y_test,y_pred))

clf = MyRandomForestClassifier(10, 28)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
