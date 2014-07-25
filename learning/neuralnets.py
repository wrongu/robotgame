import os
import numpy as np
import random
from math import exp

def rand_plus_or_minus():
	return (random.random() * 2.0) - 1.0

@np.vectorize
def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

@np.vectorize
def sigmoid_derivy(y):
	return y * (1.0 - y)

@np.vectorize
def sigmoid_deriv(x):
	return np.multiply(sigmoid(x), 1.0 - sigmoid(x))

def output_error(o, t):
	diff = t - o
	return 0.5 * diff * diff

def error_gradient(o, t):
	return t - o

def choose(x, y, bias=0.5):
	return x if random.random() < bias else y
intersperse = np.vectorize(choose)

class MultiLayerPerceptron(object):

	def __init__(self, layer_sizes=[], mats=None, biases=None, random=0.0):
		self.__W = mats or []
		self.__b = biases or []
		self.__layers = len(layer_sizes)
		self.__layer_sizes = layer_sizes
		self.__clear_vectors()
		if not mats:
			self.__W = [np.matrix(np.zeros((layer_sizes[i+1], layer_sizes[i]))) for i in xrange(self.__layers-1)]
		if not biases:
			self.__b = map(lambda s: np.zeros((s,1)), self.__layer_sizes)
		self.mutate(edit=random)

	def __clear_vectors(self):
		# biases
		self.__b = map(lambda s: np.zeros((s,1)), self.__layer_sizes)
		# sum stage (a = sigmoid(z + b))
		self.__z  = map(lambda s: np.zeros((s,1)), self.__layer_sizes)
		# activations
		self.__a = map(lambda s: np.zeros((s,1)), self.__layer_sizes)

	def copy(self):
		# using __class__ allows subclasses to be copied also
		c = self.__class__(self.__layer_sizes)
		for i in range(self.__layers - 1):
			c.__W[i] = self.__W[i].copy()
		return c

	def mutate(self, edit=0.001):
		for mat in self.__W:
			mat += np.matrix((np.random.rand(*mat.shape) * 2.0 - 1.0) * edit)
		for b in self.__b:
			b += (np.random.rand(*b.shape) * 2.0 - 1.0) * edit
		return self

	def set_input(self, idx, value):
		self.__a[0][idx] = value

	def set_inputs(self, input_vector):
		self.__a[0] = input_vector

	def get_inputs(self):
		return self.__a[0].copy()

	def propagate(self):
		for l in xrange(self.__layers-1):
			self.__z[l+1] = self.__W[l] * self.__a[l] + self.__b[l+1]
			self.__a[l+1] = sigmoid(self.__z[l+1])

	def backpropagate(self, target, learning_rate=.2):
		"""Live update of matrix weights.
			'target' close to 1.0 reinforce (and 0.0 alter) behavior
		"""
		#########
		# step 1: compute errors at each layer
		#########
		errors = [None] * self.__layers
		L = self.__layers - 1 # L is highest layer index
		# error in last layer is defined by targets
		errors[L] = np.multiply(self.__a[L] - target, sigmoid_derivy(self.__a[L]))
		# propagate error backwards through the network
		for l in range(L-1, 0, -1):
			errors[l] = np.multiply(self.__W[l].transpose() * errors[l+1], sigmoid_derivy(self.__a[l]))
		#########
		# step 2: update matrix weights and biases based on errors
		#########
		for l in range(L):
			self.__W[l] -= learning_rate * (errors[l+1] * self.__a[l].transpose())
			self.__b[l+1] -= learning_rate * errors[l+1]
		final_error = target - self.__a[L]
		final_error = np.multiply(final_error, final_error)
		return sum(final_error).item(0)

	def get_output(self, which):
		return self.__a[-1][which]

	def choose_output(self, strict=False):
		if strict:
			# strict maximum
			return np.argmax(self.__a[-1])
		else:
			# weighted random choice
			cs = self.__a[-1].cumsum()
			weight = random.random() * cs.item(-1)
			return sum(cs < weight)

	def get_output_vector(self):
		return self.__a[-1].copy()

	@classmethod
	def mate(cls, p0, p1, bias):
		assert p0.__layer_sizes == p1.__layer_sizes
		newp = cls(layer_sizes=p0.__layer_sizes)
		for l in xrange(newp.__layers - 1):
			# select each weight randomly from either parent
			newp.__W[l] = intersperse(p0.__W[l], p1.__W[l], bias)
			newp.__b[l] = intersperse(p0.__b[l], p1.__b[l], bias)
		return newp

	@classmethod
	def load(cls, file_path):
		p = None
		with open(file_path, "r") as f:
			npzdata = np.load(f)
			sizes = npzdata['arr_0']
			p = cls(sizes, random=False)
			for l in xrange(p.__layers - 1):
				p.__W[l] = np.matrix(npzdata['arr_%d' % (l + 1)])
				p.__b[l] = np.matrix(npzdata['arr_%d' % (l + p.__layers)])
		return p

	def save(self, directory, suffix=""):
		fname = "mlp_" + ("_".join([str(l) for l in self.__layer_sizes])) + suffix + ".npz"
		path = os.path.join(directory, fname)
		with open(path, "w") as f:
			mats_to_save = [np.array(self.__layer_sizes)] + self.__W + self.__b
			np.savez(f, *mats_to_save)
		return path
