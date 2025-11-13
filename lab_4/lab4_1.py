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

# THE GOAL OF THIS LAB IS TO LEARN -> DECISION_TREE_CLASSIFIER

dataset = load_wine()
X = dataset["data"]
y = dataset["target"]
feature_names = dataset["feature_names"]

#POINT 1: STUDY DATASET
print(X.shape)
print(y.shape)
print(feature_names)

print(X[np.isnan(X)]) #it's empty meaning no na values, and is result is flattened
print(Counter(y)) # how many elements per class/label

#POINT 2:
clf = DecisionTreeClassifier()
#FOR DECISION TREES IT'S NOT NECESSARY TO NORMALIZE DATA(part of preprocessing step)
#because we build the tree always considering one feature at a time, never comparing one another

clf.fit(X,y)
'''
For a Decision Tree, fit() does not use complex calculus (like Neural Networks or Linear Regression do). Instead, it performs a brute-force search known as a "Greedy Algorithm."
When you run clf.fit(X, y), the code enters a massive loop that does the following:

Step A: The Scan
The tree looks at every single feature in your dataset (Alcohol, Malic Acid, Ash, etc.) and every unique value in those columns.
    Example: It looks at "Alcohol" and asks: "What if I split the data at 12.0? What about 12.5? What about 13.0?"

Step B: The "Impurity" Check
For every one of those hypothetical splits, it calculates a score (usually Gini Impurity or Entropy).
    It asks: "If I split the wines into two groups based on 'Alcohol > 13.0', how mixed up are the groups?"
    Bad Split: Group A has 50% Class 0 and 50% Class 1. (High Impurity)
    Good Split: Group A is 100% Class 0. (Low Impurity - Pure)

Step C: The Decision
It picks the single best question that separates the classes most cleanly.
    Winner: "Flavanoids <= 1.58".
    It permanently creates a "Node" in the tree with this rule.

Step D: Recursion (Divide and Conquer)
It actually splits your data into two new piles (Left and Right) based on that rule. Then, it calls itself again on the new piles.
    "Okay, now I have this smaller pile of wines. What is the best feature to split this pile?"

When does it stop?
It repeats this process until one of three things happens:
    Purity: A node contains only one class of wine (e.g., "All wines in this pile are Class 1").
    Constraints: It hits a limit you set (e.g., max_depth=3).
    Running out of data: The node has too few samples to split further.
'''

#POINT 3: VISUALIZING THE TREE:
dot_data = export_graphviz(
    clf,                     # model
    out_file=None,           # <--- IMPORTANT: This returns the string instead of creating a file
    feature_names=feature_names,  # The names of your columns (Alcohol, etc.)
    class_names=dataset.target_names, # The names of your targets (class_0, etc.)
    filled=True,             # Colors the boxes
    rounded=True             # Makes the boxes look nicer
)

#the number of samples that reach the node, divided by class. 
# This is useful for computing the GINI index at each node. For the root node, for example, the GINI will be:
#1 - (59/178)**2 - (71/178)**2 - (48/178)**2 =
#0.6583133442747129


# Print the string so you can copy it
print(dot_data)

#POINT 4: NAIVE PREDICTION

y_pred = clf.predict(X)
print("accuracy of naive prediction: " ,accuracy_score(y,y_pred)) #100% accuracy, because we're testing model on same data used
                                #for training


#POINT 5: BETTER PREDICTION(SPLITTING DATA) 

X_train, X_test, y_train, y_test = train_test_split(
                    X,y,test_size=0.2,random_state=42,#shuffle=True IMPORTANT: if we have an imbalanced problem we might not retain the distr
                                                      #the distribution from the train label to the test label, we need to use stratify
                                                      stratify=y
)            

#POINT 6: TRAIN MODEL USING SPLIT

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("accuracy of split data set: ", accuracy_score(y_test,y_pred)) # result: 0.97
print(classification_report(y_test,y_pred))

'''
Start with the Support column on the far right.
    Definition: This is the actual number of bottles of that specific wine in your X_test (test set).

    Your Data:
        There were 12 bottles of Class 0.
        There were 14 bottles of Class 1.
        There were 10 bottles of Class 2.
        Total: 36 bottles in the test set.

2. Precision vs. Recall (The Trade-off)
These are the two most important metrics for understanding how your model makes mistakes.

Precision ("The Trustworthiness Score")
    Question: When the model claims "This is Class 0," how often is it telling the truth?
    Your Result for Class 0 (1.00): 100%. Every time your model predicted Class 0, it was actually Class 0. It never cried wolf.
    Your Result for Class 1 (0.88): 88%. When the model predicted Class 1, it was usually right, but 12% of the time it was actually looking at a different wine (Class 0 or 2) and thought it was Class 1.

Recall ("The Dragnet Score")
    Question: Out of all the Class 0 wines that actually exist, how many did the model manage to find?
    Your Result for Class 0 (0.92): 92%. It found most of them, but it missed one (1/12 is roughly 8%).
    Your Result for Class 1 (1.00): 100%. It found every single Class 1 wine in the dataset. It didn't miss a single one.

    
The reason Accuracy has only one value—while Precision, Recall, and F1-score have three (one per class)—is that Accuracy is a global score, whereas the others are local scores.
Here is the difference:

1. Accuracy = The "Final Exam Grade"
Accuracy answers the question: "Did the model get the right answer?"
It doesn't care what the answer was (Wine 0, Wine 1, or Wine 2). It just counts the total number of correct guesses and divides by the total number of bottles.

    Formula: Total BottlesTotal Correct Guesses
    In your case: You had 36 bottles. The model got roughly 34 of them right.
    36/34 = 0.94

Because it lumps all the correct answers into one big pile, you only get one single number for the entire exam.
'''

#POINT 7: MOVING AWAY FROM DEFAULT TREE, GRID SEARCH TO TUNE HYPERPARAM. AND
#         POSSIBLY REDUCE OVERFITTING
params = {
    "max_depth": [None, 2, 3, 4, 5],
    "min_impurity_decrease": [0, .01, .03, .07, .09, .11],
    "criterion": ["gini", "entropy"]

}

accuracies = []
for config in ParameterGrid(params):
    clf = DecisionTreeClassifier(**config)
    clf.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
print(max(accuracies)) #ENTROPY IMPROVES IT DRASTICALLY: 0.97(GINI) -> 1(ENTROPY)

#POINT 8: WE MIGHT BE OVERFITTING ON TEST DATA, SO WE USE KFOLDS CROSS VALIDATION TO AVOID VALIDATION SET,
#         SINCE DATA SET NOT SO BIG


X_train_valid , X_test , y_train_valid , y_test = train_test_split (X,y,test_size=0.2,random_state=42,#shuffle=True IMPORTANT: if we have an imbalanced problem we might not retain the distr
                                                      #the distribution from the train label to the test label, we need to use stratify
                                                      stratify=y)
kf = KFold(5)

accuracies = [] #accuracy per configuration 
for config in ParameterGrid(params):
    model_accuracies = [] #calculates model accuracy for each k set
    counts_k_elem = [] # keeps track of how many elements we have in each k set
    for train_indices, valid_indices in kf.split(X_train_valid):
        X_train = X_train_valid[train_indices]
        y_train = y_train_valid[train_indices]
        X_valid = X_train_valid[valid_indices]
        y_valid = y_train_valid[valid_indices]
        
        # keep track of the number of elements in each split
        counts_k_elem.append(len(train_indices)) 
        
        clf = DecisionTreeClassifier(**config)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_valid, clf.predict(X_valid))
        model_accuracies.append(acc)
    accuracies.append(np.average(model_accuracies, weights=counts_k_elem))

best_config = list(ParameterGrid(params))[np.argmax(accuracies)]
clf = DecisionTreeClassifier(**best_config)
clf.fit(X_train_valid, y_train_valid)
print(accuracy_score(y_test, clf.predict(X_test)))

print()

print(dict(zip(feature_names,clf.feature_importances_)))