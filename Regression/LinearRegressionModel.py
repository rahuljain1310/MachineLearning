import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import os
import math
import datetime

### =======================================================================================================
### Import Data from CSV AND Normailisation of Data
### =======================================================================================================

getDay = lambda M,D,Y : datetime.datetime(int(Y),int(M),int(D)).strftime("%w")

data = []
with open('Dataset/train.csv', 'r') as csvfile: 
  csvreader = csv.reader(csvfile) 
  next(csvreader, None)
  for row in csvreader:
    Month,Day,Year = row[0].split('/')
    WDay = getDay(Month,Day,Year)
    data.append((int(Month),int(Day),int(Year),int(WDay),float(row[1])))
data = np.array(data)
# np.random.shuffle(data)
X = data[:,0:-1]
Y = data[:,-1]

# plt.scatter(X[:,0],Y)
# plt.show()

Samples = Y.shape[0]
assert Samples == X.shape[0]

TrainingSamples = int(0.7*Samples)
ValidationSamples = int(0.9*Samples)

XTrain,XValidate,XTest = np.split(X, [TrainingSamples,ValidationSamples])
YTrain,YValidate,YTest = np.split(Y, [TrainingSamples,ValidationSamples]) 

assert XTrain.shape[0]+XValidate.shape[0]+XTest.shape[0] == Samples

### =======================================================================================================
### Functions, Operators
### =======================================================================================================

def getMPInverse(X,Y):
  Xt = X.T
  Xi = np.linalg.inv(np.dot(Xt,X))
  return np.dot(Xi,np.dot(Xt,Y))

def getMPInverseReg(X,Y,lbd):
  Xt = X.T
  Xi = np.linalg.inv(np.dot(Xt,X)+lbd*np.identity(X.shape[1]))
  return np.dot(Xi,np.dot(Xt,Y))

#-- Functions --
def Totalloss(Ypredict,Y):
  Error = Ypredict-Y
  return Error,np.mean(np.square(Error))

normalizeData = lambda data: data-data.mean(axis=0)/data.max(axis=0)
sigmoidData = lambda x: 1/(1+math.exp(-x))

#-- Operators --
getYpredict = lambda W,X : X.dot(W.T)
# RegGradient = lambda W: np.concatenate(([0],W[1:]))
Sign = lambda X: 1 if X > 0 else -1

def getDesignMatrix (X,Deg):
  polynomial = lambda x: [pow(x,i) for i in range(Deg+1)]
  return np.array(list(map(polynomial,X)))

def GetRMSProp(Beta,Vw,grad):
  Vw = (Beta*Vw)+(1-Beta)*np.square(grad)
  Deno = np.sqrt(Vw)+ 1e-8
  RMSprop = grad/Deno
  return Vw,RMSprop

### =======================================================================================================
### Normalization And Feature Engineering
### =======================================================================================================

XTrain_Normalized = normalizeData(XTrain)
XValidate_Normalized = normalizeData(XValidate)

#-- Feature Matrix --#
def getFeatureMatrix(DataMatrix):
  assert DataMatrix.shape[1] == 4
  # return np.array( [ [1, x1, x3, x1*x1, math.log(x1), x1*x3*x3, x1*x1*x3, x1*x1*x1*x1] for (x1,x2,x3) in DataMatrix ] )
  # return np.array( [ [1, x1, x3, x1*x1, x3*x3, x3*x3*x1, x1*x1*x1, x1*x1*x1*x1] for (x1,x2,x3) in DataMatrix ] )
  # return np.array( [ [1,x1,x3, math.cos(x1),math.cos(x3),math.sin(x1),math.sin(x3), x1*x3, x1*x1, x3*x3, x1*x1*x3,x3*x3*x1, x1*x1*x1,x3*x3*x3, x3*x3*x3*x3, x1*x3*x3*x3, x1*x1*x3*x3, x1*x1*x1*x3, x1*x1*x1*x1, x1*x1*x1*x1*x1, x3*x3*x3*x3*x3 ] for (x1,x2,x3) in DataMatrix ] )
  return np.array( [ [pow(x1,i) for i in range(0,10)]+[x4,1/(1+math.exp(-x4)),math.log(1+x1),1/(1+math.exp(-x3)), x3 , math.sqrt(x3), math.cos(x1),math.cos(x3),math.sin(x1),math.sin(x3), 1/(1+x1*x1), 1/(1+x3*x3) ,x1*x3, x3*x3*x1] for (x1,x2,x3,x4) in DataMatrix ] )

def GradientDescent (X,Y,Xc,Yc,Iterations,alpha,lbd=0,ShowGraph=False):
  M = X.shape[1]
  W = np.random.rand(M)/100                          ## Initilization of Weights, W
  # Vw = np.zeros(M,dtype=float)
  LossIteration = []
  CrossLossIteration = []
  IterationMark = int(Iterations/10) 
  for i in range(Iterations):
    if i%IterationMark==0:
      alpha = alpha*0.5
    # -- Mini Batch Gradient Descent -- 
    Yp = getYpredict(W,X)
    E,loss = Totalloss(Yp,Y)
    loss+=lbd*np.linalg.norm(W)
    grad = E.dot(X)+2*lbd*W
    # Vw,grad = GetRMSProp(0.9,Vw,grad)
    W = W-alpha*grad
    LossIteration.append(loss)
    Yp = getYpredict(W,Xc)
    E,loss = Totalloss(Yp,Yc)
    CrossLossIteration.append(loss)
  if ShowGraph:
    plt.plot(LossIteration)
    plt.plot(CrossLossIteration,color='green')
    plt.show()
  return W

### HyperParameters ###
Xm = getFeatureMatrix(XTrain_Normalized)
# Xm = np.vectorize(sigmoidData)(Xm)
XmC = getFeatureMatrix(XValidate_Normalized)
# XmC = np.vectorize(sigmoidData)(getFeatureMatrix(XValidate_Normalized))
# WOptimal = GradientDescent(Xm,YTrain,XmC,YValidate,Iterations=400000,alpha=0.004,ShowGraph = True)
MinLoss = math.inf
WOptimalMPI = None
for lbd in np.arange(0,35,0.01):
  WMPI = getMPInverseReg(Xm,YTrain,lbd)
  Yp = getYpredict(WMPI,XmC)
  error,loss = Totalloss(Yp,YValidate)
  if loss<MinLoss:
    MinLoss = loss
    WOptimalMPI = WMPI

### =======================================================================================================
### Visualization of Results on Training
### =======================================================================================================

# print("\nVisualization : Moore  Training Error\n")

Yp = getYpredict(WOptimalMPI,Xm)
error,loss = Totalloss(Yp,YTrain)
Yp = getYpredict(WOptimalMPI,XmC)
errorCross,lossCross = Totalloss(Yp,YValidate)
Yp = getYpredict(WOptimalMPI,getFeatureMatrix(normalizeData(XTest)))
errorTest,lossTest = Totalloss(Yp,YTest)

NoiseMean = np.mean(error)
NoiseDeviation = math.sqrt(np.var(error))

# print("Weights: ",WOptimalMPI)
print("Training Loss: ",loss)
print("Cross Validation Loss: ",lossCross)
print("Test Loss: ",lossTest)
# print("Noise Mean: ", NoiseMean)
# print("Noise Deviation", NoiseDeviation)

# print("\nVisualization : Gradient Descent Training Error\n")
# Yp = getYpredict(WOptimal,Xm)
# error,loss = Totalloss(Yp,YTrain)
# Yp = getYpredict(WOptimal,XmC)
# errorCross,lossCross = Totalloss(Yp,YValidate)

# NoiseMean = np.mean(error)
# NoiseDeviation = math.sqrt(np.var(error))

# print("Weights: ",WOptimal)
# print("\nTraining Loss: ",loss)
# print("Cross Validation Loss: ",lossCross)
# print("Noise Mean: ", NoiseMean)
# print("Noise Deviation", NoiseDeviation)

# plt.hist(error)
# plt.show()

### =======================================================================================================
### Visualization of Results Test
### =======================================================================================================

print("\nVisualization : Prediction Test Data")

XTest,YTest = [],[]
with open('Dataset/test.csv', 'r') as csvfile: 
  csvreader = csv.reader(csvfile) 
  next(csvreader, None)
  for row in csvreader:
    Month,Day,Year = row[0].split('/')
    WDay = getDay(Month,Day,Year)
    XTest.append((int(Month),int(Day),int(Year),int(WDay)))

XTest = np.array(XTest)
XTest_Normalized = normalizeData(XTest)
Xmt = getFeatureMatrix(XTest_Normalized)

Yp = getYpredict(WOptimalMPI,Xmt)
np.savetxt("submission.csv",Yp)
print(Yp)