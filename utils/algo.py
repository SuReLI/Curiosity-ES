import os
import sys
import numpy as np
import random 
import wandb
import json
import tensorflow as tf
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.wanbd_server import WandbServer
from utils.individual import Individual
from utils.cur_individual import Individual as Curious_Individual
# from utils.lip_individual import Individual as Curious_Individual


class Algorithm:
	def __init__(self, env, lambd=56, algo_name='curiosity_es', nb_epoch=1000,  map_size=50, seed=None, wandb_set_server=True) -> None:
		print(algo_name)
		# population
		self.population=[Individual.remote(env)  for i in range(lambd)] if algo_name!='curiosity_es' and algo_name!='aurora'  and algo_name!='curiosity_GA' and algo_name!='rnd_es' else [Curious_Individual.remote(env)  for i in range(lambd)]
		# env
		self.env=env
		# map
		self.map=np.zeros(tuple([map_size for _ in range(env.b_space)]),dtype=object)
		# archive coord
		self.archive_coord=[]
		# archive genome
		self.archive_genome=[]
		# seed
		seed=str(self.reset_seed(seed=seed))
		self.seed=seed
		# normalize ctd
		self.normalize=0.00001
		# wandb
		self.algo_name=algo_name
		self.project='TELOV2'
		self.name=algo_name+'_'+seed+'_'+os.environ['HOSTNAME']+'_'+self.env.name
		self.wandb_server=WandbServer(project_name=self.project, name=self.name) if wandb_set_server else None
		self.prefix=env.name
		# epoch
		self.nb_epoch=nb_epoch
		# check archive folder
		self.path_coord='archive_coord'+'/'+self.env.name+'/'
		os.makedirs(self.path_coord) if not os.path.exists(self.path_coord) and env.type=='maze' else True

		self.path_genome='archive_genome'+'/'+self.env.name+'/'
		os.makedirs(self.path_genome) if not os.path.exists(self.path_genome) else True
		# filename 
		self.filename='new_'+seed+'_'+os.environ['HOSTNAME']+'.json'


	def reset_seed(self, seed):
		seed=np.random.randint(100) if seed==None else seed
		# python random 
		random.seed(seed)
		# numpy 
		np.random.seed(seed)
		# tensorflow
		tf.random.set_seed(seed) 
		return seed

	def train(self) : 
		for epoch in range(self.nb_epoch):
			# process(update) specific algo 
			data, pop_coord, pop_genome =self.process(epoch)
			# update map
			for bcoord in pop_coord : self.map=self.env.update_map(self.map, bcoord)
			# coverage data
			coverage=self.map_coverage()
			# log wandb
			data[self.prefix+'/'+'epoch']=epoch
			data[self.prefix+'/'+'coverage']=coverage 
			wandb.log(data) if self.wandb_server!=None else False
			# save coords 
			self.archive_coord+=pop_coord 
			self.save_coord(epoch) 
			# # # save genome
			self.archive_genome+=pop_genome 
			self.save_genome(epoch) 
			# log
			print('coverage : ', coverage)
			print('epoch : ',epoch)
		return coverage


	def process(self, epoch):
		data={}
		pop_coord=[]
		pop_genome=[]
		return data, pop_coord, pop_genome

	def save_coord(self,k):
		if self.env.type=='maze':
			if k==0:
				with open(self.path_coord+self.filename, 'w') as outfile:
					data={'archive_coord':self.archive_coord}
					json.dump(data,outfile)
				# delete
				self.archive_coord=[]
			if k%5==0:
				# deserialize
				with open(self.path_coord+self.filename) as json_file:
					data = json.load(json_file)
				# append
				data['archive_coord']+=self.archive_coord
				# serialize
				with open(self.path_coord+self.filename, 'w') as outfile:
					json.dump(data,outfile)
				# delete
				self.archive_coord=[]
		else : 
			self.archive_coord=[]

	def save_genome(self,k):
		if k==0:
			with open(self.path_genome+self.filename, 'w') as outfile:
				data={'archive_genome':self.archive_genome}
				json.dump(data,outfile)
			# delete
			self.archive_genome=[]
		if k%10==0:
			# deserialize
			with open(self.path_genome+self.filename) as json_file:
				data = json.load(json_file)
			# append
			data['archive_genome']+=self.archive_genome
			# serialize
			with open(self.path_genome+self.filename, 'w') as outfile:
				json.dump(data,outfile)
			# delete
			self.archive_genome=[]


	def map_coverage(self):
		# coverage
		flatten=self.map.flatten()
		filled=flatten[np.where(flatten!=0)]
		size=self.map.shape[0]
		for i in range(1,len(self.map.shape)): size*=self.map.shape[i]
		percentage=100*len(filled)/size
		return percentage




if __name__=='__main__':
	from env.GymMaze.CMaze import CMaze
	from env.dm_control.Ball_in_cup import Ball_in_cup
	env=CMaze(filename='HARD')
	a=Algorithm(env)
