import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.estimator import DNNClassifier
from tensorflow.keras.datasets.mnist import load_data
import time
import random
import numpy as np
import matplotlib.pyplot as plt 
import ray 

class ICM(tf.keras.Model): 

	def __init__(self,observation_space, action_space, alpha_icm=1e-4, p= 1, m=10, beta=0.2, batch_size=8192, max_buffer=100000, gamma=0.99):
		super(ICM, self).__init__()
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
		self.beta=beta
		self.gamma=gamma
		# loss
		self.loss = tf.keras.losses.MeanSquaredError()
		# loss curiosity 
		self.loss_get_cur = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

		# Optimizer
		self.optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_icm,name='Adam_optimizer')
		### LAYERS ###
		# Feature encoder
		self.LR1=Dense(64,"relu",name="LR1")
		self.LR2=Dense(64,"relu",name="LR2")
		self.LR3=Dense(32,"linear",name="LR3")
		# Inverse 
		self.I1=Dense(32,"relu",name="Inverse_Model1")
		self.I2=Dense(action_space.shape[0],"linear",name="Inverse_Model2")
		# Forward
		self.F1=Dense(32,"relu",name="Forward_Model1")
		# has to be the same shape of phi 
		self.F2=Dense(32,"linear",name="Forward_Model2") 

		

	def feature(self,x):
		z=self.LR1(x)
		z=self.LR2(z)
		z=self.LR3(z)
		return z

	def inverse(self,z):
		i=self.I1(z)
		i=self.I2(i)
		return i

	def forward(self,z):
		f=self.F1(z)
		f=self.F2(f)
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
		for k in range(self.p):
			S1,S,A=self.sample()
			with tf.device('/CPU:0'):
				with tf.GradientTape() as tape:
					# ********LF********
					PHI_1=self.feature(S1)
					PHI=self.feature(S)
					x=np.concatenate((PHI,A),axis=1)
					PHI_1_H=self.forward(x)
					LF=self.loss(PHI_1,PHI_1_H)
					# *******LI*******
					x=np.concatenate((PHI,PHI_1),axis=1)
					Apred=self.inverse(x)
					LI=self.loss(A,Apred)
					# ******L*******
					Loss=self.beta*LF+(1-self.beta)*LI
					global_loss+=Loss
					global_lf+=LF
					global_li+=LI
					grads = tape.gradient(Loss, self.trainable_variables)
					self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		return global_loss, global_li, global_lf

	def add_buffer(self, states, actions):
		S1v,Sv,Av=[],[],[]
		S1,S,A=[],[],[]
		step_sample=round(100/self.m)
		i_sample=np.arange(1,len(states),step_sample)
		#  add buffer 
		for i in i_sample : 
			# S1
			S1.append(states[i])
			# S
			S.append(states[i-1])
			# A
			A.append(actions[i-1])
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
		# if self.S.shape[0]>self.max_buffer:
		# 	delete=self.S.shape[0]-self.max_buffer
		# 	self.S1=self.S1[delete:]
		# 	self.S=self.S[delete:]
		# 	self.A=self.A[delete:]

	def get_curiosity(self,states,actions):
		""" takes as input the trajectory and output the curiosity """
		S,A,S1=np.array(states[:-1]),np.array(actions[:-1]), np.array(states[1:])
		PHI_1=self.feature(S1)
		PHI=self.feature(S)
		xl=np.concatenate((PHI,A),axis=1)
		PHI_1_H=self.forward(xl)
		LF=self.loss_get_cur(PHI_1,PHI_1_H)
		return np.append(np.array(LF),np.array([0.0]),axis=0)
		
	


if __name__=="__main__":
	from utils.individual import Individual
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Ball_in_cup()
	env=Finger()
	print(env.observation_space.shape)
	# env=Stacker()
	# env=CMaze(filename='HARD',time_horizon=1000)
	print(env.reset().shape)
	cur=ICM(env,max_buffer=2000,batch_size=2048)
	population=[Individual.remote(env) for k in range(5)]
	for k in range(10):
		res=ray.get([i.eval.remote(env) for i in population])
		c=cur.get_curiosity(res[0]['S'], res[0]['A'])

		for r in res : cur.add_buffer(r['S'], r['A'])
	cur.sample()
	# Update
	losses=[]
	for k in range(10):
		losses.append(cur.update())
	# print("Curiosity after : ", cur.get_curiosity(np.array(states),np.array(actions)))
	plt.figure()
	plt.plot(range(10),losses)
	plt.show()
	# for i in population:
	# 	state=env.reset()
	# 	states=[]
	# 	actions=[]
	# 	done = False
	# 	while not done :
	# 		action=ray.get(i.act.remote(state))
	# 		# add buffer 
	# 		states.append(state)
	# 		actions.append(action[0])
	# 		# step
	# 		state, reward, done, info = env.step(action[0])

	# 	cur.add_buffer(np.array(states),np.array(actions))

	
	# # check prediction 
	# individual=Individual.remote(env) 
	# while not done :
	# 	action=ray.get(i.act.remote(state))
	# 	# add buffer 
	# 	states.append(state)
	# 	actions.append(action[0])
	# 	# step
	# 	state, reward, done, info = env.step(action[0])
	# # 	# render
	# # 	env.render("C")
	# # cur.a_hat(states,actions)
	

	# print("Curiosity before : ", cur.get_curiosity(np.array(states),np.array(actions)))

	# # Update
	# losses=[]
	# for k in range(10):
	# 	losses.append(cur.update())
	# print("Curiosity after : ", cur.get_curiosity(np.array(states),np.array(actions)))
	# plt.figure()
	# plt.plot(range(10),losses)
	# plt.show()


	# # check prediction 
	# while not done :
	# 	action=ray.get(i.act.remote(state))
	# 	# add buffer 
	# 	states.append(state)
	# 	actions.append(action)
	# 	# step
	# 	state, reward, done, info = env.step(action)
	# 	# render
	# 	env.render("C")
	# cur.a_hat(states,actions)

	

	
	


	
