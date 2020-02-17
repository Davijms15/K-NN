import pandas as pd
from sklearn import model_selection
import math
import numpy as np

class KNN:
    def __init__(self, neighbors = 17, loops = 100):
        self.neighbors = neighbors
        self.loops = loops

    def fit(self, X, Y):
        n_samples = X.shape[0]
        
        if self.neighbors > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")
        
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        
        self.names = list(map(str, Y[Y.columns[0]].unique()))
        self.confusion_matrix = pd.DataFrame([[0]*len(self.names) for i in range(len(self.names))], columns = self.names, index = self.names)
        self.X = X
        self.Y = Y

    def calculate_distance(self, i, j):
        distance = 0
        for k in range(len(i)):
            distance += (i[k] - j[k]) ** 2
        return math.sqrt(distance)

    def predict(self, test_size = 0.3):
        distances = list()
        predicts = list()
        for i in self.X_test.values:
            for j in self.X_train.values:
                distances.append(self.calculate_distance(i, j))
            distances = np.c_[distances, self.Y_train]
            distances = distances[distances[:,0].argsort()]
            frequency = {}
            for k in self.Y_train.values:
                frequency[k[0]] = 0
            for k in range(self.neighbors):
                frequency[distances[k][1]] += 1
            predicts.append(max(frequency, key=frequency.get))
            distances = list()
        return predicts
        

    def Hold_out(self, test_size = 0.3):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(self.X, self.Y, test_size = test_size)
        return X_train, X_test, Y_train, Y_test

    def Random_Subsampling(self):
        acuraccy = 0
        for i in range(self.loops):
            print("Test N°", i + 1)
            self.X_train, self.X_test, self.Y_train, self.Y_test = self.Hold_out(0.3)
            predicts = self.predict()
            pos = 0
            for k in self.Y_test.values:
                self.confusion_matrix[str(k[0])][str(predicts[pos])] += 1
                if(k[0] == predicts[pos]):
                    acuraccy += 1
                pos += 1
        return (acuraccy/(self.loops * len(self.Y_test))) * 100, 100 - (acuraccy/(self.loops * len(self.Y_test))) * 100

file_path = "Iris.csv"
csv = pd.read_csv(file_path)
X = csv.drop(csv.columns[-1], axis = 1)
Y = csv.drop(csv.columns[0:-1], axis = 1)
Knn = KNN(17, 100)
Knn.fit(X, Y)
Acuraccy, Error = Knn.Random_Subsampling()
print("Acuraccy = {0:.2f}%    |    Error = {1:.2f}%\n".format(Acuraccy, Error))
print(Knn.confusion_matrix)