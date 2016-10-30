# Most code from Tom Dobrow's Thesis
# "Is that a 7, or are you just happy to see me?
# Handwritten Digit Classification via Neural Networks"


# libraries for image and matrix manipulation and creation
import random
import numpy as np
from numpy import genfromtxt
from PIL import Image

# np.seterr(over='raise')

# sigmoid function
def nonlin(x):
	return 1/(1+np.exp(-x))


# derivative of sigmoid function
def nonlinPrime(x):
	return nonlin(x)*(1-nonlin(x))
	# return 1/(1+np.exp(-x))


class NeuralNetwork:

	testingAttributes = []
	testingLabels = []

	# tunable metaparameters
	learningRate = 0.00001
	iterations = 200000
	useBiases = True

	# instance variables
	numLayers = 0
	shape = None
	weights = []
	biases = []
	attributes = []
	labels = []
	guesses = []


	# layerSize is a tuple of the full shape of the network
	def __init__(self, layerSize):

		np.random.seed(1)

		self.numLayers = len(layerSize) - 1
		self.shape = layerSize

		# read in all of the trainging and testing data
		temp = genfromtxt('sound_attributes_mfcc.csv', delimiter=',')
		self.attributes = temp[:, range(temp.shape[1])]

		temp2 = genfromtxt('testing_attributes_mfcc.csv', delimiter=',')
		self.testingAttributes = temp2[:, range(temp2.shape[1])]

		temp3 = genfromtxt('sound_labels_mfcc.csv', delimiter=',')
		self.labels = np.reshape(temp3, (-1, 1))

		temp4 = genfromtxt('testing_labels_mfcc.csv', delimiter=',')
		self.testingLabels = np.reshape(temp4, (-1, 1))

		print"------------------"
		print"----- Shapes -----"
		print"------------------"

		print "Attributes"
		print self.attributes.shape
		print "Labels"
		print self.labels.shape

		for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
			self.weights.append(np.random.randn(l1, l2))

		print layerSize[0:]
		print
		print "BIASES"
		for layer in layerSize[1:]:
			self.biases.append(np.random.randn(1, layer))

		print "The shape of each weight layer:"
		for i in range(len(self.weights)):
			print self.weights[i].shape


	def Run(self):

		print self.learningRate
		errors = []
		deltas = []

		# initialize all of the error and delta matrices
		for layer in range(self.numLayers+1):
			self.guesses.append([])
			errors.append([])
			deltas.append([])

		print"------------------"
		print"----- Errors -----"
		print"------------------"

		# The meat. This is the actual network being trained
		for i in range(self.iterations):

			self.guesses[0] = self.attributes

			# calculate each layer of guesses, given the weights you already have
			for layer in range(self.numLayers):
				if(self.useBiases):
					self.guesses[layer+1] = nonlin(np.add(self.biases[layer], np.dot(self.guesses[layer], self.weights[layer])))
				else:
					self.guesses[layer+1] = nonlin(np.dot(self.guesses[layer], self.weights[layer]))

			# the trivial first layer of error and deltas
			errors[0] = self.labels - self.guesses[self.numLayers]
			deltas[0] = errors[0] * nonlinPrime(self.guesses[self.numLayers])

			# update each weight layer
			for layer in range(self.numLayers):

				errors[layer+1] = np.dot(deltas[layer], self.weights[self.numLayers-layer-1].T)
				deltas[layer+1] = errors[layer+1] * nonlinPrime(self.guesses[self.numLayers-layer-1])
				self.weights[self.numLayers-layer-1] += self.learningRate * np.dot(self.guesses[self.numLayers-layer-1].T, deltas[layer])
				if(self.useBiases):
					self.biases[self.numLayers-layer-1] += self.learningRate * np.sum(deltas[layer], axis=0)

			# print out error thus far
			if(self.iterations > 100):
				if (i % (self.iterations/100)) == (self.iterations/100 - 1):
					temp = []
					for row in errors[0]:
						temp.append(np.sum(np.abs(row)))
					print "Error:" + str(np.mean(temp, axis=0))
					NN.Test(False)


	def Test(self, printAll=True):


		# run our testing data through the network
		nextLayer = self.testingAttributes
		for layer in range(self.numLayers):
			if(self.useBiases):
				nextLayer = nonlin(np.add(self.biases[layer], np.dot(nextLayer, self.weights[layer])))
			else:
				nextLayer = nonlin(np.dot(nextLayer, self.weights[layer]))

		if(printAll):
			print "Testing Time:"
			print self.testingLabels
			print np.around(nextLayer , 0)

		numCorrect = 0
		for i in range(self.testingLabels.shape[0]):
			if self.testingLabels[i] == np.around(nextLayer , 0)[i]:
				numCorrect += 1
		print "Number Correct: "+str(numCorrect)+" out of : "+str(self.testingLabels.shape[0])



# Main Method
if __name__ == "__main__":

	NN = NeuralNetwork((1326,1326,1))
	#1326,1326
	NN.Run()
	# NN.Test()



	# print out the final weights
	# print"-------------------"
	# print"----- Weights -----"
	# print"-------------------"
	# for layer in range(len(NN.weights)):
	# 	print NN.weights[layer]
	# 	print ""
