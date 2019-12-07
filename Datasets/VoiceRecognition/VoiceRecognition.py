import csv
import numpy as np
from MLKit import ProbabilisticNeuralNetwork,KNearestNeighbour,LogisticRegression, LinearRegression, Perceptron, GaussianMixtureModal
import Operators as op

## Import Data from CSV
## =======================================================================================================

data = []
Classes = ['male','female']

with open('voice.csv', 'r') as csvfile: 
  csvreader = csv.reader(csvfile) 
  next(csvreader, None)
  for row in csvreader:
    rowX = []
    for x in row[0:-1]:
      rowX.append(float(x))
    rowX.append(Classes.index(row[-1]))
    data.append(rowX)
data = np.array(data)

np.random.shuffle(data)
X = op.stdAugData(data[:,0:-1])
Y = op.intData(data[:,-1])

Samples = X.shape[0]
TrainingSamples = int(0.8*Samples)
ValidationSamples = int(0.9*Samples)

XTrain,XValidate,XTest = np.split(X, [TrainingSamples,ValidationSamples])
YTrain,YValidate,YTest = np.split(Y, [TrainingSamples,ValidationSamples]) 

assert XTrain.shape[0]+XValidate.shape[0]+XTest.shape[0] == Samples

## Parzan Window Method
## =======================================================================================================

# print("\nWorking With Parzan Window..")

# parzanNetwork = ProbabilisticNeuralNetwork()
# parzanNetwork.Train(XTrain,YTrain,Classes)
# parzanNetwork.OptimizeSigma(XValidate,YValidate,np.arange(1e-2,2e-1,1e-2))

# print("Training Set")
# parzanNetwork.getPNNLoss(XTrain,YTrain,True)
# print("Cross Validation Set")
# parzanNetwork.getPNNLoss(XValidate,YValidate,True)
# print("Test Set")
# parzanNetwork.getPNNLoss(XTest,YTest,True)

## K-Nearest Neighbour
## =======================================================================================================

# print("\nWorking With k-Nearest Neighbour")

# XTrain1 = np.concatenate((XTrain,XValidate),axis=0)
# YTrain1 = np.concatenate((YTrain,YValidate),axis=0)

# knearest = KNearestNeighbour(5)
# knearest.Train(XTrain1,YTrain1,Classes)

# print("Training Set")
# knearest.getKNNLoss(XTrain1,YTrain1,True)
# # print("Cross Validation Set")
# # knearest.getKNNLoss(XValidate,YValidate,True)
# print("Test Set")
# knearest.getKNNLoss(XTest,YTest,True)

## Linear Regression
## =======================================================================================================

# print("\nWorking With Linear Regression...\n")

# lr = LinearRegression()
# lr.Train(XTrain,YTrain,10000)

# print("Training Set")
# lr.getLoss(XTrain1,YTrain1,True)
# # print("Cross Validation Set")
# # lr.getLoss(XValidate,YValidate,True)
# print("Test Set")
# lr.getLoss(XTest,YTest,True)

## Logistic Regression
## =======================================================================================================

# print("\nWorking With Logistic Regression...\n")

# lr = LogisticRegression()
# lr.Train(XTrain,YTrain,10000,classes=Classes)

# print("Training Set")
# lr.getLoss(XTrain1,YTrain1,True)
# # print("Cross Validation Set")
# # lr.getLoss(XValidate,YValidate,True)
# print("Test Set")
# lr.getLoss(XTest,YTest,True)

## Perceptron Algorithm
## =======================================================================================================

# print("\nWorking With Perceptron Algorithm...\n")

# pr = Perceptron(40)
# pr.Train(XTrain,YTrain)

# print("Training Set")
# pr.getLoss(XTrain1,YTrain1,True)
# print("Test Set")
# pr.getLoss(XTest,YTest,True)


## PCA
## =======================================================================================================

# performing preprocessing part 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt

# sc = StandardScaler() 
# pca = PCA(n_components = 2) 

# XTrain = sc.fit_transform(X) 
# XTrain = pca.fit_transform(XTrain) 

# explained_variance = pca.explained_variance_ratio_ 

# classifier = LogisticRegression(random_state = 0) 
# classifier.fit(XTrain, Y) 

# # y_pred = classifier.predict(XTest) 
# # cm = confusion_matrix(YTest, y_pred) 

# X_set, y_set = XTrain, Y 
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01) ) 
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
# 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
# 			cmap = ListedColormap(('yellow', 'white', 'aquamarine'))) 
# plt.xlim(X1.min(), X1.max()) 
# plt.ylim(X2.min(), X2.max()) 
# for i, j in enumerate(np.unique(y_set)): 
# 	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
# 				c = ListedColormap(('red', 'green', 'blue'))(i), label = j) 

# plt.title('Logistic Regression (Training set)') 
# plt.xlabel('PC1') # for Xlabel 
# plt.ylabel('PC2') # for Ylabel 
# plt.legend() # to show legend 
# plt.show() 
