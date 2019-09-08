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
### Gradient Descent Training Model
### =======================================================================================================

def GradientDescentLAD (Xm,Y,M,Iterations,alpha,ShowGraph=False):
  # MinLoss = math.inf
  W = 5*np.random.rand(M+1)                              ## Initilization of Weights, W
  Vw = np.zeros(M+1,dtype=float)
  LossIteration = []
  loss = math.inf
  mark = int(Iterations/10)
  for i in range(Iterations):
    if i%mark==0:
      alpha = alpha*0.33
    Yp = getYpredict(W,Xm)
    E,loss = Totalloss(Yp,Y)
    sign = np.vectorize(Sign)(E)
    LossIteration.append(loss)      
    grad = sign.dot(Xm)
    Vw,grad = GetRMSProp(0.9,Vw,grad)
    W = W-alpha*grad
  if ShowGraph:
    plt.plot(LossIteration[int(Iterations/5):])
    plt.show()
  return W

### HyperParameters ###
M = 8
Xm = getDesignMatrix(X,M)                       ## Get Polynomial Features
alpha = 0.001
WOptimal = GradientDescentLAD(Xm,Y,M,1000000,0.002,ShowGraph = True)
error, loss = Totalloss(getYpredict(WOptimal,Xm),Y)

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