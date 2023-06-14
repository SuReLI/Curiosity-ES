from utils.algo import Algorithm
from utils.cur_individual import Individual
from utils.non_dominated_sort import sortNondominated
from utils.icm_opt import ICM
import ray
import numpy as np
import random


class Curiosity_GA(Algorithm):

	def __init__(self, env, lambd=56, mu=7,  N=2000, sigma=0.1, map_size=50 , beta=0.9,alpha_icm=1e-4, p=32, phi=0.8, gamma=0.99, m=50, batch_size=128, max_buffer=int(5*1e5) ) :
		super().__init__(env, lambd, algo_name='curiosity_GA', nb_epoch=N, map_size=map_size)
		self.sigma=sigma
		self.lambd=lambd #size of the initial population
		self.mu=mu #number of offspring per individual
		self.genome_shape=ray.get(self.population[0].genome.remote()).shape
		self.beta=beta
		self.parent=[]
		self.archive=[]
		self.normalize=1e-4
		self.icm=ICM(env, alpha_icm=alpha_icm, p= p, m=m, beta=beta, gamma = gamma,batch_size=batch_size, max_buffer=max_buffer)
		


	def process(self, epoch):
		# evaluation
		res=ray.get([i.eval.remote(self.env) for i in self.population])
		# fis
		self.get_curiosity(res)
		# fes
		fes=list(map(lambda r : r['value'], res))
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		# selection
		self.selection(res+self.parent)
		# mutation
		self.mutation()
		# train 
		self.icm.update()
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



	def selection(self, population):
		fes=list(map(lambda r : r['value'], population))
		fis=list(map(lambda r : r['fi'], population))
		mu_fes, std_fes= np.mean(fes), np.std(fes) 
		mu_fis, std_fis= np.mean(fis), np.std(fis) 
		std_fes+=self.normalize
		std_fis+=self.normalize
		population.sort(key=lambda r : self.beta*((r['value']-mu_fes)/std_fes)+(1-self.beta)*((r['fi']-mu_fis)/std_fis))
		# fitness=list(zip(fes,fis))
		# front_index=sortNondominated(fitness)
		# self.parent=[]
		# i_front=0
		# while len(self.parent)<self.lambd:
		# 	for index in front_index[i_front]: self.parent.append(population[index])
		# 	i_front+=1
		self.parent=population[:self.lambd]

	def mutation(self):
		offspring=[]
		for p in self.parent: 
			genome=p['genome']
			offspring+=[genome+np.random.normal(0, self.sigma, self.genome_shape) for _ in range(int(self.lambd/self.mu))]
		# set offrpring to population 
		ray.get([i.set_genome.remote(gen) for (i,gen) in zip(self.population,offspring)])
	
	def get_curiosity(self, res):
		for r in res : 
			# fi
			fi,fi_reg,z=self.icm.get_curiosity(r['S'], r['A'])
			r['fi']=fi
			self.icm.add_buffer(r['S'], r['A'])
		
	


if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Stacker()
	env=CMaze(filename='SNAKE')
	algo=Curiosity_GA(env)
	# for k in range(30):
	# 	algo.process(k)
	# 	print(k)
	algo.train()