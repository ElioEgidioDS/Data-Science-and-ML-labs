import numpy as np
import pandas as df
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from KNearestNeighbours import KNearestNeighbors


def accuracy_score(y_true, y_pred):
    return (y_true==y_pred).sum()/len(y_true)

data_path = r'C:\Users\UTENTE\ames_housing_dataset\AmesHousing.csv'
data = df.read_csv(data_path, sep=',')

#DEFINE TARGET VARIABLE TO "HOUSE STYLE" 
y = np.array(data["House Style"])

# SELECT ONLY NUMERICAL FEATURES
data = data.select_dtypes( include = np.number )


#REPLACE MISSING VALUES WITH MEAN OF THE FEATURE
data.fillna(data.mean(axis=0), inplace=True)

X = data.values

#SEPARATE INTO TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,shuffle=True
)

#FIRST EVALUATION USING KNN 
knn_model = KNearestNeighbors(5, "euclidean", "uniform")
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("accuracy of knn, first eval: ", accuracy_score(y_test, y_pred)) # 0.61, very low


#COMPARE IT WITH JUST GUESSING THE MOST LABEL VALUE
unique, counts = np.unique(y_test, return_counts=True)
random_guess = counts.max() / len(y_test)
print("random guess", random_guess) #too close to our model, we must improve it

#IMPROVING OUR MODEL -> scale of features matters a lot, we must rescale them by standardizing them
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

#IMPORTANT: WE MUST USE TRAIN MEAN AND STD ON TEST SET AS WELL, if not DATA LEAKAGE occurs
X_train_std = (X_train - mean) / std
X_test_std = (X_test - mean) / std

knn_model = KNearestNeighbors(5, "euclidean", "uniform")
knn_model.fit(X_train_std, y_train)
y_pred = knn_model.predict(X_test_std)
print("accuracy of knn, second(std) eval: ", accuracy_score(y_test, y_pred)) # 0.84, way better

#----------------WE STUDY WHICH K IS BETTER-------------------------------

neighs = range(1, 20)
perf_uniform = np.empty(len(neighs))

for i, n_neigh in enumerate(neighs):
    knn_model = KNearestNeighbors(n_neigh, "euclidean", "uniform")
    knn_model.fit(X_train_std, y_train)
    perf_uniform[i] = accuracy_score(y_test, knn_model.predict(X_test_std))

'''fig, ax = plt.subplots()
ax.plot(neighs, perf_uniform)
plt.show() #K = 10 SEEMS BEST CHOICE'''


''' #----------------USING FOLDS----------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
perf = np.empty((5, len(neighs)))

for run, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    for i, n_neigh in enumerate(neighs):
        knn_model = KNearestNeighbors(n_neigh, "euclidean", "uniform")
        knn_model.fit(X_train_std, y_train)
        perf[run, i] = accuracy_score(y_test, knn_model.predict(X_test_std))

perf_mean = perf.mean(axis=0)
plt.plot(neighs, perf_mean)
plt.show() #NOW BEST CHOICE IS K = 5

#------------------------------------------------------------'''

# HYPERPARAMETERS EVALUATION THROUGH ITERATION OF K'S AND DISTANCE METRIC
neighs = range(1, 20)
metrics = ["euclidean", "manhattan"]

perf = np.empty((len(neighs), len(metrics)))

for i, n_neigh in enumerate(neighs):
    for j, metric in enumerate(metrics):
        knn_model = KNearestNeighbors(n_neigh, metric, "uniform")
        knn_model.fit(X_train_std, y_train)
        perf[i, j] = accuracy_score(y_test, knn_model.predict(X_test_std))

fig, ax = plt.subplots()
cm = ax.imshow(perf)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=90)
fig.colorbar(cm)
plt.show()