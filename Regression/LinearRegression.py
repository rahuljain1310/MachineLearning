import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import math

### =======================================================================================================
### Import Data from CSV
### =======================================================================================================

from numpy import genfromtxt
Data = genfromtxt('Gaussian_noise.csv', delimiter=',')
X,Y = Data[:,0],Data[:,1]
N = X.shape[0]

plt.scatter(X,Y)
plt.show()

## Conclusion :-
## Odd Function With 2 Maxima and 2 Minima - 5 Degree Function 

### =======================================================================================================
### Functions, Operators
### =======================================================================================================

#-- Functions --
def Totalloss(Ypredict,Y):
  Error = Ypredict-Y
  return Error,np.sum(np.square(Error))/len(Y)

#-- Operators --
getYpredict = lambda W,X : X.dot(W.T)
RegGradient = lambda W: np.concatenate(([0],W[1:]))

def getDesignMatrix (X,Deg):
  polynomial = lambda x: [pow(x,i) for i in range(Deg+1)]
  return np.array(list(map(polynomial,X)))

### =======================================================================================================
### Moore Inverse
### =======================================================================================================

def getMPInverse(X,Y):
  Xt = X.T
  Xi = np.linalg.inv(np.dot(Xt,X))
  return np.dot(Xi,np.dot(Xt,Y))

## Applying Moore Inverse on Each Value of M
LossM = []
for i in range(1,10):
  Xm = getDesignMatrix(X,i)
  W = getMPInverse(Xm,Y)
  Yp = getYpredict(W,Xm)
  Error,loss = Totalloss(Yp,Y)
  LossM.append(loss)

## Visualization of LossM
plt.plot(LossM)
plt.show()

#-- Conclusion:- 
#-- From Moore Inverse We get that the best fit lies for M = 5

M = 5
Xm = getDesignMatrix(X,M)
W = getMPInverse(Xm,Y)
Yp = getYpredict(W,Xm)
Error,loss = Totalloss(Yp,Y)

print(W,loss)

plt.scatter(X,Yp,cmap="red")
plt.scatter(X,Y,cmap="yellow")
plt.show()

plt.hist(Error)
plt.show()

### =======================================================================================================
### Gradient Descent Training Model
### =======================================================================================================

### HyperParameters ###
M = 5
Iterations = 2000
alpha = 0.002
Lamda = 0.001

def GradientDescent (Xm,Y,M,BatchSize,ShowGraph=False):
  W = np.random.rand(M+1)/1000                            ## Initilization of Weights, W
  TotalBatch = math.ceil(Xm.shape[0]/BatchSize)           ## Get Total Batches To divide Into
  XBatch = np.array_split(Xm,TotalBatch)                  ## Create Batches
  YBatch = np.array_split(Y,TotalBatch)
  LossIteration = []

  for i in range(Iterations):
    for j in range(TotalBatch):
      # -- Mini Batch Gradient Descent -- 
      Yp = getYpredict(W,XBatch[j])
      E,_ = Totalloss(Yp,YBatch[j])
      grad = E.dot(XBatch[j]) # + 2*RegGradient(W)
      W = W - alpha*grad
      # -- Calculating Loss After Each Iteration -- 
      Yp = getYpredict(W,Xm)
      _,loss = Totalloss(Yp,Y)
      LossIteration.append(loss)

  if ShowGraph:
    plt.plot(LossIteration)
    plt.show()

  return W

MinLoss = math.inf
Error = None   
WOptimal = None
Xm = getDesignMatrix(X,M)                       ## Get Polynomial Features

for batchSize in range(100,101):
  W = GradientDescent(Xm,Y,M,batchSize,False)
  error, loss = Totalloss(getYpredict(W,Xm),Y)
  if(loss<MinLoss):
    OptimalW = W
    MinLoss = loss
    Error = error


### =======================================================================================================
### Visualization of Noise
### =======================================================================================================

print(W,MinLoss)
plt.hist(Error,bins=20)
plt.show()