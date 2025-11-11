import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from KNearestNeighbours import KNearestNeighbors

def accuracy_score(y_true, y_pred):
    return (y_true==y_pred).sum()/len(y_true)

iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header = None)
data_path = r'C:\Users\UTENTE\ames_housing_dataset\AmesHousing.csv'
data = pd.read_csv(data_path, sep=',')
mnist_data = pd.read_csv("mnist_test.csv",header= None,sep=',')


'''WRONG: print(data.loc('House Style')) 
because loc use [] and requires both row and column
CORRECT: data.loc[:, 'House Style']
'''
print(data["House Style"].unique().tolist())
#value_counts(drop_na = false) when exploring the data, we want to know how complete dataset is
print(data["House Style"].value_counts(dropna=False))
#no missing values



#IRIS KNN:
X = iris_data.iloc[:,:4].values
y = iris_data.iloc[:,-1].values

X_m = mnist_data.values[:,1:].astype(float)
y_m = mnist_data.values[:,0].astype(int)

X_100 = np.vstack([ X_m[y_m==d][:100] for d in range(10) ])
y_100 = np.hstack([ [d]*100 for d in range(10) ])

X_m_train, X_m_test, y_m_train, y_m_taste = train_test_split(
    X_100,y_100,test_size=0.2,random_state=42,shuffle=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, random_state=42, shuffle=True
)
#X_train is now a numpy array, if i wanted to give the columns names i would need to turn it
#into a df
#column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
#X_test_df = pd.DataFrame(X_test, columns=column_names)

knn_model = KNearestNeighbors(5, "euclidean", "uniform")
knn_model.fit(X_m_train, y_m_train)
y_m_pred = knn_model.predict(X_m_test)

print(accuracy_score(y_m_taste, y_m_pred))