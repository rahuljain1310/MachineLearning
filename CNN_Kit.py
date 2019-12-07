import cv2
import numpy as np
from CNN_Kit import relu, pooling, conv
from os import listdir
from os.path import isfile, join


### ========= Layer Units ========== ###

class FCLayer():
  def __init__(self, fltshape):
    self.weights = np.zeros(fltshape[0], fltshape[1]+1)
    self.y = None

  def forwardPass(self,x):
    x = np.concatenate(np.ones(1),x)
    self.y = np.matmul(self.weights, x)
  
  def backWardPass(self, e, fltGrid):
    error = np.matmul(self.weights.T,e)
    gradient = None
    fltGrid += gradient
    return error

  def update(self, gradient, m):
    self.filter += gradient/m


class CNNLayer():
  def __init__(self, fltshape):
    self.filter = np.zeros(fltshape)
    self.filterOut = None
    self.pool = None
    self.relu = None

  def forwardPass(self,x):
    self.filterOut = conv(x,self.filter)
    self.pool = pool(self.filterOut)
    self.relu = relu(self.pool)
    return self.relu

  def backWardPass(self, e, fltGrad):
    self.poolError = None
    self.filterOutError = None
    self.inputError = None
    gradient = None
    fltGrad += gradient
    return self.inputError

  def update(self, gradient, m):
    self.filter += gradient/m
  
### ========= Designing CNN Architecture ========== ###

def CNN_Modal():
  def __init__(self):
    self.layer1 = CNNLayer((4,3,3,3))
    self.layer2 = CNNLayer((3,3,3,4))
    self.layer3 = CNNLayer((1,3,3,3))
    self.layer4 = FCLayer((10,25))
    self.layer5 = FCLayer((4,10))

    self.gradientLayer1 = np.zeros(self.layer1.filter.shape)
    self.gradientLayer2 = np.zeros(self.layer2.filter.shape)
    self.gradientLayer3 = np.zeros(self.layer3.filter.shape)
    self.gradientLayer4 = np.zeros(self.layer4.filter.shape)
    self.gradientLayer5 = np.zeros(self.layer5.filter.shape)

  def getLoss(p,y):
    return p-y

  def forwardPass(self,x):
    out1 = self.layer1.forwardPass(x)
    out2 = self.layer2.forwardPass(out1)
    out3 = self.layer3.forwardPass(out2)
    out4 = self.layer4.forwardPass(out3)
    out5 = self.layer5.forwardPass(out4)
    return out5

  def backWardPass(self,L5):
    L4 = self.layer5.backWardPass(L5,self.gradientLayer5)
    L3 = self.layer4.backWardPass(L4,self.gradientLayer4)
    L2 = self.layer3.backWardPass(L3,self.gradientLayer3)
    L1 = self.layer2.backWardPass(L2,self.gradientLayer2)
    L0 = self.layer1.backWardPass(L1,self.gradientLayer1)

  def TrainMiniBatch(self,mb):
    m = len(mb)
    print("Received Training Batch of Size {0}".format(m))
    for inputImage in mb:
      x,y = inputImage
      _,x = cv2.imread(x)
      cv2.imshow('fr',x)
      p = self.forwardPass(inputImage)
      L = self.getLoss(p,y)
      self.backWardPass(L)
    self.layer1.update(self.gradientLayer1,m)
    self.layer2.update(self.gradientLayer2,m)
    self.layer3.update(self.gradientLayer3,m)
    self.layer4.update(self.gradientLayer4,m)
    self.layer5.update(self.gradientLayer5,m)


# if __name__ == "__main__":
#   classes = ['Stop', 'Next', 'Previous', 'Background']
#   images1 = [(join('Test_Images/Stop',f),0) for f in listdir('Test_Images/Stop') if f.endswith('.jpg')]
#   images2 = [(join('Test_Images/Next',f),1) for f in listdir('Test_Images/Next') if f.endswith('.jpg')]
#   images3 = [(join('Test_Images/Previous',f),2) for f in listdir('Test_Images/Previous') if f.endswith('.jpg')]
#   images4 = [(join('Test_Images/Background',f),3) for f in listdir('Test_Images/Background') if f.endswith('.jpg')]
#   images = images1+images2+images3+images4
#   cnn = CNN_Modal()
#   cnn.TrainMiniBatch(images[0:64])

  