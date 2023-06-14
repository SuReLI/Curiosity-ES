import numpy as np 
from sklearn.preprocessing import normalize

class NeuralNetNumpy: 
	def __init__(self) -> None:
		self.layers={}
	
	def dense(self,layer):
		self.layers[layer.name]=layer

	@property
	def genome(self):
		genome=[]
		for name,layer in self.layers.items(): 
			kernel=layer.W.flatten()
			genome.append(kernel)
			bias=layer.B.flatten()
			genome.append(bias)
		return np.concatenate(genome,axis=0)

	@genome.setter
	def genome(self, genome):
		count=0
		for name,layer in self.layers.items(): 
			# shape
			kernel_shape=layer.W.shape
			bias_shape=layer.B.shape
			# Kernel weights from genome
			kernel_weights=genome[count:count+kernel_shape[0]*kernel_shape[1]]
			# reshape kernel 
			kernel=np.reshape(kernel_weights,kernel_shape)
			# set weights
			self.layers[name].W=kernel
			# add kernel shape index
			count+=kernel_shape[0]*kernel_shape[1]
			# Bias weights from genome
			bias_weights=genome[count:count+bias_shape[0]]
			# reshape bias
			bias=np.reshape(bias_weights,bias_shape)
			# set weights
			self.layers[name].B=bias
			# add bias shape index
			count+=bias_shape[0]
			
class LipNeuralNetNumpy: 
	def __init__(self) -> None:
		self.layers={}
	
	def dense(self,layer):
		self.layers[layer.name]=layer

	@property
	def genome(self):
		genome=[]
		for name,layer in self.layers.items(): 
			kernel=layer.W.flatten()
			genome.append(kernel)
			bias=layer.B.flatten()
			genome.append(bias)
		return np.concatenate(genome,axis=0)

	@genome.setter
	def genome(self, genome):
		count=0
		for name,layer in self.layers.items(): 
			# shape
			kernel_shape=layer.W.shape
			bias_shape=layer.B.shape
			# Kernel weights from genome
			kernel_weights=genome[count:count+kernel_shape[0]*kernel_shape[1]]
			# reshape kernel 
			kernel=np.reshape(kernel_weights,kernel_shape)
			# normalise
			kernel=normalize(kernel,axis=1,norm='l2')*0.5
			# set weights
			self.layers[name].W=kernel
			# add kernel shape index
			count+=kernel_shape[0]*kernel_shape[1]
			# Bias weights from genome
			bias_weights=genome[count:count+bias_shape[0]]
			# reshape bias
			bias=np.reshape(bias_weights,bias_shape)
			# set weights
			self.layers[name].B=bias
			# add bias shape index
			count+=bias_shape[0]

class NumpyLayer:
	def __init__(self,input,unit, activation ,name="NumpyLayer") -> None:
		self.W=np.zeros((unit,input))
		self.B=np.zeros((unit,1))
		self.name=name
		if activation=='sigmoid':
			self.activation=sigmoid
		elif activation=='relu':
			self.activation=relu
		elif activation=='softmax':
			self.activation=softmax
		elif activation=='linear':
			self.activation=linear
		else: 
			raise( 'No activation function defined')

	def __call__(self, x) :
		return self.activation(self.W@x+self.B)
		
		
def sigmoid(Z):
	return 1/(1+np.exp(-Z))

def relu(Z):
	return np.maximum(0,Z)

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0) # only difference

def linear(x):
	return x

def tanh(x):
	return np.tanh(x)