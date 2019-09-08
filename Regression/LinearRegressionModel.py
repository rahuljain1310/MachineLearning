import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import os
import math

### =======================================================================================================
### Import Data from CSV
### =======================================================================================================

Data = genfromtxt('NonGaussian_noise.csv', delimiter=',')
X,Y = Data[:,0],Data[:,1]
N = X.shape[0]

plt.scatter(X,Y)
plt.show()

### =======================================================================================================
### Functions, Operators
### =======================================================================================================

#-- Functions --
def Totalloss(Ypredict,Y):
  Error = Ypredict-Y
  return Error,np.mean(np.square(Error))

#-- Operators --
getYpredict = lambda W,X : X.dot(W.T)
RegGradient = lambda W: np.concatenate(([0],W[1:]))
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

### =======================================================================================================
### Conclusion, Results and Visualization
### =======================================================================================================

#-- From Moore Inverse We get that the best fit lies for M = 5

print("Visualization : Moore Penrove.")

M = 8
Xm = getDesignMatrix(X,M)
W = getMPInverse(Xm,Y)
Yp = getYpredict(W,Xm)
Error,loss = Totalloss(Yp,Y)
NoiseMean = np.mean(Error)
NoiseDeviation = math.sqrt(np.var(Error))

print("Weights: ",W)
print("Loss: ",loss)
print("Noise Mean: ", NoiseMean)
print("Noise Deviation", NoiseDeviation)

plt.scatter(X,Yp,cmap="red")
plt.scatter(X,Y,cmap="yellow")
plt.show()

plt.hist(Error)
plt.show()

### =======================================================================================================
### Gradient Descent Training Model
### =======================================================================================================

def GradientDescentLAD (Xm,Y,M,Iterations=2000,alpha=0.001,ShowGraph=False):
  MinLoss = math.inf
  OptimalW = None
  for _ in range(1):
    W = 5*np.random.rand(M+1)                              ## Initilization of Weights, W
    Vw = np.zeros(M+1,dtype=float)
    LossIteration = []
    loss = math.inf
    for i in range(Iterations):
      Yp = getYpredict(W,Xm)
      E,loss = Totalloss(Yp,Y)
      sign = np.vectorize(Sign)(E)
      LossIteration.append(loss)      
      grad = sign.dot(Xm)
      # Vw,grad = GetRMSProp(0.9,Vw,grad)
      W = W-alpha*grad

    if loss<MinLoss:
      MinLoss = loss
      OptimalW = W
      if ShowGraph:
        plt.plot(LossIteration)
        plt.show()

  return OptimalW

### HyperParameters ###
M = 8
Xm = getDesignMatrix(X,M)                       ## Get Polynomial Features
alpha = 0.001
Lamda = 0.001
WOptimal = GradientDescentLAD(Xm,Y,M,Iterations=500000,ShowGraph = False)
error, loss = Totalloss(getYpredict(W,Xm),Y)

### =======================================================================================================
### Visualization of Noise
### =======================================================================================================

print("Visualization : Gradient Descent.")

Yp = getYpredict(WOptimal,Xm)
error,loss = Totalloss(Yp,Y)

NoiseMean = np.mean(error)
NoiseDeviation = math.sqrt(np.var(error))

print("Weights: ",WOptimal)
print("Loss: ",loss)
print("Noise Mean: ", NoiseMean)
print("Noise Deviation", NoiseDeviation)


plt.scatter(X,Yp,cmap="red")
plt.scatter(X,Y,cmap="yellow")
plt.show()

plt.hist(error)
plt.show()