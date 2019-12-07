import csv
import numpy as np
import math
from scipy import stats
import Operators as op

## Functions ##
def GradientDescent(X,Y,getYpredict,getGradient,Iterations,alpha=0.001):
  D = X.shape[1]
  W = np.random.rand(D)
  for _ in range(Iterations):
    Yp = getYpredict(W,X)
    grad = getGradient(Yp,Y,X)
    W -= alpha*grad
  return W

class ProbabilisticNeuralNetwork():
  def __init__(self):
    self.W = None
    self.patternCategoryMatrix = None
    self.optimizedSigma = 0.5
    self.classes = None
    self.ParzanWindow = lambda x,sigma : math.exp((x-1)/(sigma*sigma))

  def getPatternCategoryMatrix(self,Y):
    TrainingSamples = Y.shape[0]
    patternCategoryMatrix = np.zeros((TrainingSamples, len(self.classes)))
    patternCategoryMatrix[np.arange(TrainingSamples),Y] = 1
    return patternCategoryMatrix

  def Train(self,XTrain,YTrain,classes):
    self.classes = classes
    self.W = op.NormalizePattern(XTrain).T
    self.patternCategoryMatrix = self.getPatternCategoryMatrix(YTrain)

  def OptimizeSigma(self, XValidate, YValidate, rge):
    minLoss = math.inf
    for sigma in rge:
      loss,_ = self.getPNNLoss(XValidate,YValidate)
      if loss<minLoss:
        self.optimizedSigma = sigma
        minLoss = loss

  def getProbabilityMatrix(self,X,sigma):
    X = op.NormalizePattern(X)
    s = np.vectorize(self.ParzanWindow)(X.dot(self.W),sigma)
    s = s.dot(self.patternCategoryMatrix)
    s = op.getProbability(s)
    return s

  def getPNNresult(self,X,sigma):
    s = self.getProbabilityMatrix(X,sigma)
    return op.classify(s)

  def getPNNLoss(self,X,Y,printLoss = False):
    s = self.getProbabilityMatrix(X,self.optimizedSigma)
    result = op.classify(s)
    rm = np.zeros(s.shape)
    rm[np.arange(Y.shape[0]),Y] = 1
    loss = np.linalg.norm(s-rm,axis=1)
    loss = loss.mean()
    Wrong = np.count_nonzero(Y-result)
    Accuracy = 100.0-(Wrong*100.0)/Y.shape[0]
    if printLoss:
      print("Loss: {0}, WrongPredictions: {1}, Accuracy: {2}%".format(loss,Wrong,Accuracy))
    else:
      return loss,Wrong

class KNearestNeighbour():
  def __init__(self,K):
    self.W = None
    self.ClassVector = None
    self.classes = None
    self.K = K

  def Train(self,XTrain,YTrain,classes):
    self.W = XTrain
    self.ClassVector = YTrain
    self.classes = classes

  def getNearestNeighbour(self,X):
    s = np.linalg.norm(self.W-X,axis=1)
    nearest = np.argsort(s)[0:self.K]
    return s[nearest], self.ClassVector[nearest]

  def getKNNresult(self,X):
    Result = []
    for x in X:
      _,result = self.getNearestNeighbour(x)
      Result.append(stats.mode(result)[0][0])
    Result = np.array(Result)
    return Result

  def getKNNLoss(self,X,Y,printLoss = False):
    result = self.getKNNresult(X)
    Wrong = np.count_nonzero(Y-result)
    Accuracy = 100.0-(Wrong*100.0)/Y.shape[0]
    if printLoss:
      print("WrongPredictions: {0}, Accuracy: {1}%".format(Wrong,Accuracy))
    else:
      return Wrong,Accuracy

class LinearRegression():
  def __init__(self):
    self.W = None
    self.getYpredict = lambda W,X : X.dot(W.T)
    self.getGradient = lambda Yp,Y,X : (Yp-Y).dot(X)
    self.squareloss = lambda Yp,Y : np.mean(np.square(Yp-Y))

  def Train(self,XTrain,YTrain,iterations,alpha=1e-4):
    self.W = GradientDescent(XTrain,YTrain,self.getYpredict,self.getGradient,iterations,alpha=alpha)
  
  def getresult(self,X):
    return self.getYpredict(self.W,X)

  def getLoss(self,X,Y,printLoss = False):
    Yp = self.getYpredict(self.W,X)
    loss = np.mean(np.square(Yp-Y))
    if printLoss:
      print("Square Loss: {0}".format(loss))
    else:
      return loss
  
class LogisticRegression():
  def __init__(self):
    self.W = None
    self.getYpredict = lambda W,X : op.SigmoidVector(X.dot(W.T))
    self.getGradient = lambda Yp,Y,X : (Yp-Y).dot(X)

  def crossEntropyLoss(self,Yp,Y):
    loss = 0
    N = Y.shape[0]
    for i in range(Y.shape[0]):
      y,yh = Y[i],Yp[i]
      if(y==1):
        loss -= math.log(yh)
      else:
        loss -= math.log(1-yh)
    return loss/N

  def Train(self,XTrain,YTrain,iterations, classes = None):
    self.W = GradientDescent(XTrain,YTrain,self.getYpredict,self.getGradient,iterations,alpha=1E-4)
  
  def getresult(self,X):
    return self.getYpredict(self.W,X)

  def getLoss(self,X,Y,printLoss = False):
    Yp = self.getYpredict(self.W,X)
    result = np.vectorize(lambda x: x > 0.5)(Yp)
    Wrong = op.getWrong(result,Y)
    Accuracy = op.getAccuracy(Wrong,Y.shape[0])
    # loss = self.crossEntropyLoss(Yp,Y)
    if printLoss:
      print("WrongPredictions: {0}, Accuracy: {1}%".format(Wrong,Accuracy))
    else:
      return Wrong,Accuracy

class Perceptron():
  def __init__(self, threshold=100):
      self.threshold = threshold
      
  def predict(self, inputs):
    summ = np.matmul(inputs, self.W)
    return summ > 0

  def Train(self, X, labels):
    D = X.shape[1]
    self.W = np.random.rand(D)
    for _ in range(self.threshold):
      for inputs, label in zip(X, labels):
        prediction = self.predict(inputs)
        if ( not prediction and label == 1):
            self.W = self.W + inputs
        if ( prediction and label == 0):
            self.W = self.W - inputs
  
  def getresult(self,X):
    result = X.dot(self.W)
    result = np.vectorize(lambda x: x > 0.5)(result)
    return result
  
  def getLoss(self,X,Y,printLoss=False):
    result = self.getresult(X)
    Wrong = np.count_nonzero(Y-result)
    Accuracy = 100.0-(Wrong*100.0)/Y.shape[0]
    if printLoss:
      print("WrongPredictions: {0}, Accuracy: {1}%".format(Wrong,Accuracy))
    else:
      return Wrong,Accuracy

class Softmax():
  def __init__(self,classes):
    self.W = None
    self.classes = classes

  def getPatternCategoryMatrix(self,Y):
    TrainingSamples = Y.shape[0]
    patternCategoryMatrix = np.zeros((TrainingSamples, len(self.classes)))
    patternCategoryMatrix[np.arange(TrainingSamples),Y] = 1
    return patternCategoryMatrix

  def getLoss(self,X,Y,printLoss=False):
    Yc = self.getPatternCategoryMatrix(Y)
    A=X.dot(self.W)
    B=np.exp(A)
    ans=np.trace(np.transpose(A).dot(Yc))-np.sum(np.log(np.sum(B,axis=1)))
    ans=ans/X.shape[0]
    loss = (-1)*ans

    s = X.dot(self.W)
    result = op.classify(s)
    Wrong = np.count_nonzero(Y-result)
    Accuracy = 100.0-(Wrong*100.0)/Y.shape[0]
    if printLoss:
      print("Square Loss: {0}, WrongPredictions: {1}, Accuracy: {2}%".format(loss,Wrong,Accuracy))
    else:
      return Wrong,Accuracy

  def getGradient(self,W,X,Y):
    Y_hat=np.exp(X.dot(W))
    tempsum=np.sum(Y_hat,axis=1)
    Y_hat=Y_hat/tempsum.reshape([Y_hat.shape[0],1])
    A=Y_hat-Y
    d=(np.transpose(X)).dot(A)
    d=d/X.shape[0]
    return d

  def Train(self,X,Y,step,itr):
    Y = self.getPatternCategoryMatrix(Y)
    self.W=np.zeros((X.shape[1],Y.shape[1]))
    for _ in range(itr):
      self.W -= step*self.getGradient(self.W,X,Y)

  def getPredictions(self,X,W):
    s = X.dot(self.W)
    result = op.classify(s)
    return result

class GaussianMixtureModal():
  def __init__(self, XTrain,N = 2, Iteration = 5):
    D = XTrain.shape[1]
    self.totalGaussian = N
    self.count = 0
    self.Gprob = np.repeat(1/N,N)
    self.Gprob = self.Gprob / np.sum(self.Gprob)
    self.Gmeans = np.random.rand(N,D)/10
    self.GStd = np.array([np.identity(D,dtype=float)]*N)
    self.Train(XTrain,Iteration)

  
  def getGaussian(self):
    return self.Gprob, self.Gmeans, self.GStd

  def getGaussianProb(self,G,x):
    mean,prob,std = G
    # print(std,np.linalg.det(std))
    return prob*stats.multivariate_normal.pdf(x,mean=mean,cov=std)

  def getSampleProb(self,x,gaussians):
    llSample = 0
    for G in gaussians:
      llSample += self.getGaussianProb(G,x)
    return llSample

  def getLogLikelihood(self,X,gaussians):
    logLikelihood = 0
    for x in X:
      logLikelihood += math.log(self.getSampleProb(x,gaussians))
    return logLikelihood/X.shape[0]

  def getGammaEstimate(self,gaussians,X):
    N = X.shape[0]
    probData = np.zeros((N,self.totalGaussian))
    for i in range(N):
      for j in range(self.totalGaussian):
        probData[i][j] = self.getGaussianProb(gaussians[j],X[i])
    probData = (probData.T)/np.sum(probData,axis=1)
    return probData.T

  def getLatentParameter(self,X):
    gaussians = list(zip(self.Gmeans,self.Gprob,self.GStd))
    prob = self.getGammaEstimate(gaussians,X)
    return np.argmax(prob,axis=1)

  def Train(self,X,iterations):
    if self.count == 5:
      print("Maximum Attempts Exceeded.")
      return
    N = X.shape[0]
    D = X.shape[1]
    ll = 0
    try:
      for i in range(iterations):
        gaussians = list(zip(self.Gmeans,self.Gprob,self.GStd))
        ll = self.getLogLikelihood(X,gaussians)
        print("Iteration No: {0} Log Likelihood: {1}".format(i,ll), end='\r', flush=True)
        ## E - Step ##
        gamma = self.getGammaEstimate(gaussians,X)
        gammaSum = gamma.sum(axis=0)

        ## M - Step ##

        mean = np.matmul(gamma.T,X)
        Covariance = []
        for j in range(self.totalGaussian):
          std = np.zeros((D,D),dtype=float)
          u,_,_ = gaussians[j] 
          for k in range(N):
            Xn = np.array([X[k]-u])
            XnMatrix = np.matmul(Xn.T,Xn)
            std += gamma[k][j]*XnMatrix
          std = std/gammaSum[j]
          Covariance.append(std)
        Covariance = np.array(Covariance)

        ## Updating Standard Deviation 
        self.GStd = Covariance
        for j in range(self.totalGaussian):
          mean[j] = mean[j]/gammaSum[j]
        self.Gmeans = mean
        self.Prob = gammaSum/N   
      print("Training Complete LogLikelihood Final Value: {0}".format(ll))
    except: 
      self.Train(X,iterations) 
      self.count += 1
      