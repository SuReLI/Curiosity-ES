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

class ICM(tf.keras.Model): 

	def __init__(self,env, alpha_icm=1e-4, p= 5, m=10, beta=0.2, batch_size=2048, max_buffer=10000, gamma=0.99, dim_enc=32):
		super(ICM, self).__init__()
		self.observation_space, self.action_space=env.observation_space, env.action_space
		# buffer
		# self.S=[]
		# self.S1=[]
		# self.A=[]
		self.start=0
		# percentage of the trajectory
		self.max_buffer=max_buffer
		self.p=p
		self.m=m
		self.batch_size=batch_size
		self.beta=beta
		self.gamma=gamma
		# loss
		self.loss = tf.keras.losses.MeanSquaredError()
		# Optimizer
		self.optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_icm,name='Adam_optimizer')
		### LAYERS ###
		
		self.F1=Dense(32,"relu",name="Forward_Model1")
		self.F2=Dense(32,"relu",name="Forward_Model2")
		self.F3=Dense(self.observation_space.shape[0],"linear",name="Forward_Model3") 
		


	def forward(self,z):
		f=self.F1(z)
		f=self.F2(f)
		f=self.F3(f)
		return f


	def sample(self):
		sample_id=np.random.randint(0,self.S1.shape[0],min(self.batch_size,self.S1.shape[0]))
		S1=self.S1[sample_id]
		S=self.S[sample_id]
		A=self.A[sample_id]
		# n_S1,n_S,n_A=zip(*random.sample(list(zip(self.S1,self.S,self.A)),min(self.batch_size,len(self.S))))
		# S1,S,A=np.concatenate(n_S1,axis=0),np.concatenate(n_S,axis=0),np.concatenate(n_A,axis=0)
		# print(n_S1)
		return S1,S,A

	def update(self):
		global_loss=0
		global_li=0
		global_lf=0
		global_lrec=0
		for k in range(self.p):
			S1,S,A=self.sample()
			with tf.device('/CPU:0'):
				with tf.GradientTape() as tape:
					# ********LF********
					x=np.concatenate((S,A),axis=1)
					S_1_H=self.forward(x)
					LF=self.loss(S1,S_1_H)
					Loss=LF
					global_loss+=Loss
					grads = tape.gradient(Loss, self.trainable_variables)
					# self.optimizer2.apply_gradients(zip(grads, self.trainable_variables))
					self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

		return global_loss

	def add_buffer(self, states, actions):
		S1v,Sv,Av=[],[],[]
		S1,S,A=[],[],[]
		step_sample=round(100/self.m)
		i_sample=np.arange(1,len(states),step_sample)
		# i_sample=random.sample(range(1, len(states)), step_sample)
		#  add buffer 
		for i in i_sample : 
			# S1
			S1.append(states[i])
			# S
			S.append(states[i-1])
			# A
			A.append(actions[i-1])
		# # S1
		# self.S1.append(np.array(S1))
		# # S
		# self.S.append(np.array(S))
		# # A
		# self.A.append(np.array(A))
		# S1
		S1v.append(np.array(S1))
		# S
		Sv.append(np.array(S))
		# A
		Av.append(np.array(A))
		# concatenate 
		S1v=np.concatenate(S1v,axis=0)
		Sv=np.concatenate(Sv,axis=0)
		Av=np.concatenate(Av,axis=0)
		if not self.start: 
			self.S1=S1v
			self.S=Sv
			self.A=Av
			self.start=True
		else:
			self.S1=np.concatenate((self.S1,S1v),axis=0) 
			self.S=np.concatenate((self.S,Sv),axis=0) 
			self.A=np.concatenate((self.A,Av),axis=0) 
		if self.S.shape[0]>self.max_buffer:
			delete=self.S.shape[0]-self.max_buffer
			remove_by_index= random.sample(range(0, self.S.shape[0]), delete)
			self.S1=np.delete(self.S1,remove_by_index, axis=0)
			self.S=np.delete(self.S,remove_by_index, axis=0)
			self.A=np.delete(self.A,remove_by_index, axis=0)
			# self.S1=self.S1[delete:]
			# self.S=self.S[delete:]
			# self.A=self.A[delete:]


			

	def get_curiosity(self,states,actions):
		""" takes as input the trajectory and output the curiosity """
		S,A,S1=np.array(states[:-1]),np.array(actions[:-1]), np.array(states[1:])
		# ********LF********
		x=np.concatenate((S,A),axis=1)
		S_1_H=self.forward(x)
		LF=self.loss(S1,S_1_H,sample_weight=self.gamma**np.arange(S_1_H.shape[0],0,-1))
		# LREC
		# LREC=self.loss(S1,self.decode(PHI_1_H),sample_weight=self.gamma**np.arange(PHI_1_H.shape[0],0,-1))
		# LREC=self.loss(S1,self.decode(PHI_1 + tf.exp(0.5 * STD_1) * tf.random.normal(shape=(tf.shape(PHI_1)[0],tf.shape(PHI_1)[1]))),sample_weight=self.gamma**np.arange(PHI_1_H.shape[0],0,-1))
		return LF
		
	


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
	cur=ICM(env,max_buffer=2000,batch_size=64, p=32)
	population=[Individual.remote(env) for k in range(5)]
	# for k in range(5):
	res=ray.get([i.eval.remote(env) for i in population])
	# print(res[0]['S'])
	# print(res[0]['A'])
	c=cur.get_curiosity(res[0]['S'], res[0]['A'])
	for r in res : cur.add_buffer(r['S'], r['A'])
	# # cur.sample()
	# # # Update
	global_losses=[]
	global_lfs=[]
	global_lis=[]

	for k in range(10):
		global_loss= cur.update()
		global_losses.append(global_loss)
		
	# print("Curiosity after : ", cur.get_curiosity(np.array(states),np.array(actions)))
	
	plt.figure()
	plt.plot(range(10),global_losses, label='Loss')
	# plt.plot(range(10),global_lis, label='LI')
	# plt.plot(range(10),global_lfs, label='LF')
	plt.legend()
	plt.show()