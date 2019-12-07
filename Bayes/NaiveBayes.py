# Naive Bayes On The Iris Dataset

from math import sqrt, exp, pi
import csv
import numpy as np

### ====================== Naive Bayes ==================================== ###

mean = lambda samples : sum(samples)/float(len(samples)) 
stDev = lambda samples : sqrt(sum([(x-mean(samples))**2 for x in samples]) / float(len(samples)-1)) 
Gaussianprobability = lambda x,u,s : (1/(sqrt(2*pi)*s))*exp(-((x-u)**2/(2 * s**2 )))
accuracy = lambda ground, pred : 100*np.mean(ground==pred) 

class NaiveBayes():

	def __init__(self):
		self.prior = dict()
		self.likelihood = dict()
		self.classes = list()
		
	def Segregation_By_Class(self, X, Y):
		DataClassWise = dict()
		for class_ in self.classes: 
			if (class_ not in DataClassWise): DataClassWise[class_] = list()
		for sample, class_ in zip(X,Y): 
			DataClassWise[self.classes[class_]].append(sample)
		return DataClassWise
	
	def ModelParameters(self, dataset):
		likelihood = [(mean(feature), stDev(feature)) for feature in zip(*dataset)]
		return likelihood, len(dataset)
		
	def Train(self, X, Y, classes):
		self.classes = classes
		DataClassWise = self.Segregation_By_Class(X, Y)
		for _class, class_samples in DataClassWise.items():
			self.likelihood[_class], self.prior[_class] = self.ModelParameters(class_samples)

	def getClassProbabilities(self,s):
		probabilities = dict()
		for _class, class_distributions in self.likelihood.items():
			probabilities[_class] = self.prior[_class]
			for i in range(len(class_distributions)):
				mean, stdev = class_distributions[i]
				probabilities[_class] *= Gaussianprobability(s[i], mean, stdev)
		return probabilities
	
	def predictSample(self,s):
		probabilities = self.getClassProbabilities(s)
		return max(probabilities, key=probabilities.get)
 
	def getPredictions(self,testData):
		predictions = [self.predictSample(testSample) for testSample in testData]
		return predictions
	
	def evaluate(self, XTest, YTest):
		Ypred = self.getPredictions(XTest)
		Ypred = np.array([self.classes.index(res) for res in Ypred])
		accuracy =  100*np.mean(Ypred==YTest)
		print('Mean Accuracy: %.3f%%' % accuracy)

### ===================== Import Data from CSV ============================ ###

def LoadIrisData():
	data = []
	with open('Iris.csv', 'r') as csvfile: 
		csvreader = csv.reader(csvfile) 
		next(csvreader, None)
		for row in csvreader:
			rowX = list()
			for x in row[0:-1]: rowX.append(float(x))
			rowX.append(Classes.index(row[-1]))
			data.append(rowX)
	data = np.array(data)
	X = data[:,1:-1]
	Y = data[:,-1].astype(int)
	return X,Y

if __name__ == "__main__":

	Classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
	X,Y = LoadIrisData()
	Samples = X.shape[0]
	TrainingSamples = int(0.8*Samples)
	XTrain, XTest = np.split(X, [TrainingSamples])
	YTrain, YTest = np.split(Y, [TrainingSamples]) 

	nb = NaiveBayes()
	nb.Train(XTrain, YTrain, Classes)
	nb.evaluate(XTest, YTest)