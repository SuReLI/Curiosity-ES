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

#Specific to Aurora
class AE(tf.keras.Model):     
	def __init__(self,env, alpha=1e-4, m=40, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, max_train_epoch=100):
		super(AE, self).__init__()
		self.observation_space, self.action_space=env.observation_space, env.action_space
		# buffer
		self.ReplayBuffer=np.array([])
		self.start=0
		# percentage of the trajectory
		self.max_buffer=max_buffer
		self.cross_eval = cross_eval
		self.m=m
		self.min_average_evolution=min_average_evolution
		self.ema_beta_d=ema_beta_d
		self.max_train_epoch=max_train_epoch
		# loss
		self.loss = tf.keras.losses.MeanSquaredError()
		# Optimizer
		self.optimizer=tf.keras.optimizers.Adam(learning_rate=alpha,name='Adam_optimizer')
		### LAYERS ###
		# Feature encoder
		self.E1=Dense(512,"relu",name="E1")
		self.E2=Dense(128,"relu",name="E2")
		self.E3=Dense(32,"linear",name="E3")
		# decoder 
		self.D1=Dense(32,"relu",name="D1")
		self.D2=Dense(128,"relu",name="D2")
		self.D3=Dense(512,"relu",name="D3")
		self.D4=Dense(int(10**(-2)*m*env.time_horizon*(self.observation_space.shape[0]+self.action_space.shape[0])),"linear",name="D4")

	def encoder(self,x):
		z=self.E1(x)
		z=self.E2(z)
		z=self.E3(z)
		return z

	def decoder(self,z):
		x=self.D1(z)
		x=self.D2(x)
		x=self.D3(x)
		x=self.D4(x)
		return x

	def sample(self):
		sample_id=np.random.randint(0,self.ReplayBuffer.shape[0],min(self.batch_size,self.ReplayBuffer.shape[0]))
		batch=self.ReplayBuffer[sample_id]
		return batch

	def shuffle_along_axis(self, a, axis):
		idx = np.random.rand(*a.shape).argsort(axis=axis)
		return np.take_along_axis(a,idx,axis=axis)

	def increase(self, cross_val_scores):
		if len(cross_val_scores)<self.min_average_evolution:
			return False
		else : 
			d_ema=0
			for i in range(1,len(cross_val_scores)):
				d_ema=self.ema_beta_d*d_ema+ (1-self.ema_beta_d)*(cross_val_scores[i]-cross_val_scores[i-1])
			return True if d_ema>0 else False


	def update(self):
		cross_val_scores= []
		train_epoch=0
		while not self.increase(cross_val_scores) or train_epoch<self.max_train_epoch:
			cross_val_score=0
			for k in range(self.cross_eval):
				# shuffle
				self.ReplayBuffer = self.shuffle_along_axis(self.ReplayBuffer, axis=0)
				# cross val eval 
				id_split=int(0.75*self.ReplayBuffer.shape[0]) #75 %
				train_batch, test_batch= self.ReplayBuffer[:id_split], self.ReplayBuffer[id_split:]
				train_loss, test_loss= self.update_on_cross(train_batch, test_batch)
				cross_val_score+=-test_loss
				train_epoch+=1
			cross_val_scores.append(cross_val_score)
		return cross_val_score





	def update_on_cross(self, train_batch, test_batch):
		with tf.GradientTape() as tape:
			# train
			z=self.encoder(train_batch)
			x_hat=self.decoder(z)
			train_loss=self.loss(train_batch,x_hat)
			grads = tape.gradient(train_loss, self.trainable_variables)
			self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
			# test
			z=self.encoder(test_batch)
			x_hat=self.decoder(z)
			test_loss=self.loss(test_batch,x_hat)
		return train_loss, test_loss

	def get_BNDR(self, states, actions):
		Sv,Av=[],[]
		S,A=[],[]
		step_sample=round(100/self.m)
		i_sample=np.arange(0,len(states),step_sample)
		#  add buffer 
		for i in i_sample : 
			S.append(states[i])
			A.append(actions[i])
		Sv.append(np.array(S))
		Av.append(np.array(A))
		# concatenate 
		Sv=np.concatenate(Sv,axis=0)
		Av=np.concatenate(Av,axis=0)
		BNDR=np.concatenate((Sv.flatten(),Av.flatten()),axis=0)
		return BNDR
		
	def add_buffer(self, BNDR):
		if not self.start: 
			self.ReplayBuffer=np.expand_dims(BNDR,axis=0)
			self.start=True
		else:
			self.ReplayBuffer=np.concatenate((self.ReplayBuffer,np.expand_dims(BNDR,axis=0)),axis=0) 
		if self.ReplayBuffer.shape[0]>self.max_buffer:
			delete=self.ReplayBuffer.shape[0]-self.max_buffer
			self.ReplayBuffer=self.ReplayBuffer[delete:]
		return np.expand_dims(BNDR,axis=0)

	def get_BDR(self,BNDR):
		return self.encoder(BNDR)


	def reset_replay_buffer(self):
		self.started=False
		self.ReplayBuffer=np.array([])

if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	from utils.individual import Individual
	env=Stacker()
	# env=Finger()
	# env=Ball_in_cup()
	# env=CMaze(filename='SNAKE')
	ae=AE(env,max_buffer=2000)
	population=[Individual.remote(env) for k in range(5)]
	res=ray.get([i.eval.remote(env) for i in population])
	for r in res : BNDR=ae.add_buffer(r['S'], r['A'])
	# ae.sample()
	# print(ae.get_BDR(BNDR).shape)
	print(ae.update())
	# losses=[]
	# for k in range(10):
	# 	losses.append(ae.update())
	# # print("Curiosity after : ", cur.get_curiosity(np.array(states),np.array(actions)))
	# plt.figure()
	# plt.plot(range(10),losses)
	# plt.show()
	# print(env.action_space.shape)
	# print(env.observation_space.shape)
	# print(int(z_dim))
	# print(BNDR.shape)

