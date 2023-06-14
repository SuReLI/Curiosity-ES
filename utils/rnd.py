import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.estimator import DNNClassifier
from tensorflow.keras.datasets.mnist import load_data
import tensorflow_probability as tfp
import time
import random
import numpy as np
import matplotlib.pyplot as plt 
import ray 


class RND(tf.keras.Model): 

	def __init__(self,env, alpha_rnd=1e-4, p= 5, m=10, batch_size=2048, max_buffer=10000, gamma=0.99):
		super(RND, self).__init__()
		self.observation_space, self.action_space=env.observation_space, env.action_space
		# buffer
		# self.S=np.array([])
		# self.S1=np.array([])
		# self.A=np.array([])
		self.start=0
		# percentage of the trajectory
		self.max_buffer=max_buffer
		self.p=p
		self.m=m
		self.batch_size=batch_size
		self.gamma=gamma
		# loss
		self.loss = tf.keras.losses.MeanSquaredError()
		# Optimizer
		self.optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_rnd,name='Adam_optimizer')
		### LAYERS ###
		# Random network
		self.RN_1=Dense(32,"relu",name="RN_1", trainable=False)
		self.RN_2=Dense(32,"relu",name="RN_2", trainable=False)
		self.RN_3=Dense(1,"linear",name="RN_3", trainable=False) 
		# learning network
		self.TN_1=Dense(32,"relu",name="TN_1")
		self.TN_2=Dense(32,"relu",name="TN_2")
		self.TN_3=Dense(1,"linear",name="TN_3") 
		

		

	

	def forward(self,z):
		f=self.TN_1(z)
		f=self.TN_2(f)
		f=self.TN_3(f)
		return f
	
	def random_network(self,z):
		rn=self.RN_1(z)
		rn=self.RN_2(rn)
		rn=self.RN_3(rn)
		return rn

	def sample(self):
		sample_id=np.random.randint(0,self.S.shape[0],min(self.batch_size,self.S.shape[0]))
		S=self.S[sample_id]
		# n_S1,n_S,n_A=zip(*random.sample(list(zip(self.S1,self.S,self.A)),min(self.batch_size,len(self.S))))
		# S1,S,A=np.concatenate(n_S1,axis=0),np.concatenate(n_S,axis=0),np.concatenate(n_A,axis=0)
		# print(n_S1)
		return S

	def update(self):
		global_loss=0
		for k in range(self.p):
			S=self.sample()
			with tf.device('/CPU:0'):
				with tf.GradientTape() as tape:
					Loss=self.loss(self.random_network(S),self.forward(S))
					global_loss+=Loss
					grads = tape.gradient(Loss, self.trainable_variables)
					self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return global_loss

	def add_buffer(self, states):
		Sv=[]
		S=[]
		step_sample=round(100/self.m)
		i_sample=np.arange(1,len(states),step_sample)
		#  add buffer 
		for i in i_sample : 
			# S
			S.append(states[i])
		# S
		Sv.append(np.array(S))
		Sv=np.concatenate(Sv,axis=0)
		if not self.start: 
			self.S=Sv
			self.start=True
		else:
			self.S=np.concatenate((self.S,Sv),axis=0) 
		if self.S.shape[0]>self.max_buffer:
			delete=self.S.shape[0]-self.max_buffer
			# remove_by_index= random.sample(range(0, self.S.shape[0]), delete)
			# self.S1=np.delete(self.S1,remove_by_index, axis=0)
			# self.S=np.delete(self.S,remove_by_index, axis=0)
			# self.A=np.delete(self.A,remove_by_index, axis=0)
			self.S=self.S[delete:]
	# def add_buffer(self,s):


			

	def get_curiosity(self,states):
		""" takes as input the trajectory and output the curiosity """
		S=np.array(states)
		Loss=self.loss(self.random_network(S),self.forward(S))
		return Loss
		
	


if __name__=="__main__":
	from utils.cur_individual import Individual
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Ball_in_cup()
	env=Finger()
	# env=Stacker()
	# env=CMaze(filename='HARD',time_horizon=1000)
	cur=RND(env,max_buffer=2000,batch_size=2048)
	population=[Individual.remote(env) for k in range(5)]
	res=ray.get([i.eval.remote(env) for i in population])
	for r in res: 
		c=cur.get_curiosity(r['S'])
		print(c)
	