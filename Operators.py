import numpy as np
import math

## ==========================================================================================
## Data Manipulation
## ==========================================================================================

## All data values are in the range 0 and 1 ( Normalization of Features )
normalizeFeatures = lambda data: (data-data.min(axis=0))/data.max(axis=0)

## All data values are in the range -1 to 1
standardizeFeatures = lambda data: (data-data.mean(axis=0))/data.max(axis=0)

## Add 1 to each column -- Matrix Dimension NxD
augmentData = lambda X:  np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

## Normalize Pattern
NormalizePattern = lambda X: (X.T/np.linalg.norm(X,axis=1)).T

## Standardise and Augment Data 
stdAugData = lambda X: augmentData(standardizeFeatures(X))

## Convert Strings to Intger
intData = lambda X: np.vectorize(lambda x: int(x))(X)
floatData = lambda X: np.vectorize(lambda x: float(x))(X)

## Divide By Sum to get Probabilities
getProbability = lambda s : ((s.T)/(s.sum(axis=1)+1e-12)).T

## Classify
classify = lambda s: s.argmax(axis=1)

## Parzan Window
GaussianParzenWindow = lambda x,sigma : math.exp((x-1)/(sigma*sigma))

## Square Loss for Gradient Descent
SquareLoss = lambda Yp,Y : np.mean(np.square(Yp-Y))

## Sigmoid OPeration on Array/Elements
sigmoid = lambda x: 1/(1+math.exp(-x))
SigmoidVector = lambda X: np.vectorize(sigmoid)(X)

## For Accuracy and Loss
getWrong = lambda Y,result: np.count_nonzero(Y-result)
getAccuracy = lambda Wrong,N: 100.0-(Wrong*100.0)/N

