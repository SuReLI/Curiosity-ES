import os
import sys
from utils.evonump import NumpyLayer,NeuralNetNumpy
import ray 
import json
import numpy as np

@ray.remote
class Individual:
	""" Deterministic Policy """
	def __init__(self,env,genome=[],std=0.001):
		# super(Individual, self).__init__()
		self.env=env
		observation_shape, action_shape=env.observation_space.shape, env.action_space.shape
		### LAYERS ###
		self.nn=NeuralNetNumpy()
		self.nn.dense(NumpyLayer(input=observation_shape[0],unit=32, activation='relu',name="LR1"))
		self.nn.dense(NumpyLayer(input=32,unit=16, activation='relu',name="LR2"))
		# self.nn.dense(NumpyLayer(input=32,unit=16, activation='relu',name="LR3"))
		self.nn.dense(NumpyLayer(input=16,unit=env.action_space.shape[0], activation='linear',name="action"))
		self.smooth_tanh=1.0
		self.w=5.0 if env.type=='maze' else 1.0
		if len(genome)>0:
			self.nn.genome=genome
		else: 
			self.nn.genome=np.random.normal(0,std,self.nn.genome.shape[0])
		
	def __call__(self, x):
		x = self.nn.layers['LR1'](x.T)
		x = self.nn.layers['LR2'](x)
		# x = self.nn.layers['LR3'](x)
		action=np.tanh(self.nn.layers['action'](x)/self.smooth_tanh)*self.w
		# action = self.nn.layers['action'](x)
		return action

	def act(self,s):
		s=np.expand_dims(s, axis=0)
		action=self(s)
		return action

	def genome(self):
		return self.nn.genome
	
	def set_genome(self, genome):
		self.nn.genome=genome
		
	def eval(self,env):
		# reset
		s=env.reset()
		# done
		done=False
		# reward
		value=0
		# rollout evaluation
		while not done: 
			# act
			a=self.act(s)
			a=a.flatten() #update for icm storage		
			# step
			s, reward, done, info = env.step(a)
			# value updated
			value+=reward
			# bcoord
			b=info['bcoord']
		return {"genome": self.nn.genome,"value": value,'s':s, 'bcoord':b }


	def save(self,name='genome'):
		data={'genome':list(self.nn.genome)}
		with open('/checkpoints/'+name+'.json', 'w') as f:
			json.dump(data, f)

	def load(self,name='genome'):
		with open('/checkpoints/'+name+'.json') as f:
			data = json.load(f)
			self.nn.genome=np.array(data['genome'])

	def terminate(self):
		print("Self-killing")
		# ray.actor.exit_actor()
		# os._exit(0)

	
	

if __name__=="__main__":
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Ball_in_cup()
	# env=Finger()
	# env=Stacker()
	env=CMaze(filename='US',time_horizon=1000, n_beams=0)
	i=Individual.remote(env)
	# print(ray.get(i.genome.remote()).shape)
	print(env.reset().shape)
	# # ray.get(i.eval.remote(env))
	# # i=Individual(env)
	# # res=i.eval()
	# s=env.reset()
	# a=ray.get(i.act.remote(s))
	# print(a)
	# s,r,d,i=env.step(a)

	# print(s[0]/env.width/2)
	# print(s)
	# print(env.bcoord(s))


	