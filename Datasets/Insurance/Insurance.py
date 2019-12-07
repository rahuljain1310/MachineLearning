import csv
import numpy as np
from MLKit import ProbabilisticNeuralNetwork,KNearestNeighbour,LogisticRegression, LinearRegression, Perceptron, GaussianMixtureModal
import Operators as op

### =======================================================================================================
### Import Data from CSV
### =======================================================================================================


data = []
with open('insurance.csv', 'r') as csvfile: 
  csvreader = csv.reader(csvfile) 
  next(csvreader, None)
  for row in csvreader:
    rowX = []
    for x in row:
      rowX.append(float(x))
    data.append(rowX)
data = np.array(data)

data = np.asarray(data,dtype=float)
np.random.shuffle(data)
X = op.stdAugData(data[:,0:-1])
Y = data[:,-1]

Samples = X.shape[0]
print(Samples)
TrainingSamples = int(0.8*Samples)
ValidationSamples = int(0.8*Samples)

XTrain,XValidate,XTest = np.split(X, [TrainingSamples,ValidationSamples])
YTrain,YValidate,YTest = np.split(Y, [TrainingSamples,ValidationSamples]) 

assert XTrain.shape[0]+XValidate.shape[0]+XTest.shape[0] == Samples

### =======================================================================================================
### Linear Regression
### =======================================================================================================

print("\nWorking With Linear Regression...\n")

lr = LinearRegression()
lr.Train(XTrain,YTrain,100000,alpha=1e-5)

print("Training Set")
lr.getLoss(XTrain,YTrain,True)
# print("Cross Validation Set")
# lr.getLoss(XValidate,YValidate,True)
print("Test Set")
lr.getLoss(XTest,YTest,True)
