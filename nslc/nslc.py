from utils.algo import Algorithm
from utils.individual import Individual
from utils.non_dominated_sort import sortNondominated
import ray
import numpy as np
import random


class Nslc(Algorithm):

	def __init__(self, env, lambd=56, mu=7,  N=2000, sigma=0.00001, map_size=50 , beta=0.9, knn=30) :
		super().__init__(env, lambd, algo_name='nslc_new', nb_epoch=N, map_size=map_size)
		self.sigma=sigma
		self.lambd=lambd #size of the initial population
		self.mu=mu #number of offspring per individual
		self.genome_shape=ray.get(self.population[0].genome.remote()).shape
		self.beta=beta
		self.parent=[]
		self.archive=[]
		self.knn=knn
		


	def process(self, epoch):
		# evaluation
		res=ray.get([i.eval.remote(self.env) for i in self.population])
		# fis
		fis=[self.get_novelty(r,res+self.parent) for r in res+self.parent]
		# fes
		fes=list(map(lambda r : r['value'], res))
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		# selection
		self.selection(res+self.parent)
		# mutation
		self.mutation()
		# data
		data={
			self.prefix+'/'+'fe_max' : np.max(fes),
			self.prefix+'/'+'fe_mean' : mu_fes,
			self.prefix+'/'+'fe_std' : std_fes,
		}
		# pop_coord
		pop_coord=list(map(lambda r : r['bcoord'], res))
		# pop_genome
		pop_genome=list(map(lambda r : list(r['genome']), list(filter(lambda r : r['value'] > 1 ,res))))
		print('fe_max : ', np.max(fes))
		return data, pop_coord, pop_genome


	def get_novelty(self, r, archive):
		b = r['s']
		if len(archive)==0:
			return 0.0
		archive_b=[ i['s'] for i in archive]
		# novelty score
		b_A=np.array(archive_b)
		b_D=np.repeat(np.expand_dims(b,axis=0), len(archive_b), axis=0)
		b_S=np.sqrt(np.sum(np.square(b_A-b_D),axis=1))
		B_A=list(zip(list(b_S),archive))
		sorted(B_A, key=lambda x : x[0])
		local_niche=B_A[:self.knn]
		fes_local=list(map(lambda i : i[1]['value'], local_niche))
		dist_local=list(map(lambda i : i[0], local_niche))
		rank=self.knn #init rank 
		for f in fes_local : rank-=1 if r['value']<f else 0
		fi=np.sum(dist_local)
		r['fi']=fi
		r['rank']=rank
		return fi

	def selection(self, population):
		fes=list(map(lambda r : r['rank'], population))
		fis=list(map(lambda r : r['fi'], population))
		# mu_fes, std_fes= np.mean(fes), np.std(fes) 
		# mu_fis, std_fis= np.mean(fis), np.std(fis) 
		# population.sort(key=lambda r : self.beta*((r['value']-mu_fes)/std_fes)+(1-self.beta)*((r['fi']-mu_fis)/std_fis))
		fitness=list(zip(fes,fis))
		front_index=sortNondominated(fitness)
		self.parent=[]
		i_front=0
		while len(self.parent)<self.lambd:
			for index in front_index[i_front]: self.parent.append(population[index])
			i_front+=1
		self.parent=self.parent[:self.lambd]

	def mutation(self):
		offspring=[]
		for p in self.parent: 
			genome=p['genome']
			offspring+=[genome+np.random.normal(0, self.sigma, self.genome_shape) for _ in range(int(self.lambd/self.mu))]
		# set offrpring to population 
		ray.get([i.set_genome.remote(gen) for (i,gen) in zip(self.population,offspring)])
	

	


if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Stacker()
	env=CMaze(filename='SNAKE')
	algo=Nslc(env)
	# for k in range(30):
	# 	algo.process(k)
	# 	print(k)
	algo.train()