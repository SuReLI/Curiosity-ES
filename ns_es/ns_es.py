from utils.canonical import Canonical
from utils.algo import Algorithm
from utils.individual import Individual
import ray
import numpy as np


class NS_es(Algorithm):

	def __init__(self, env, lamb=56, N=2000, sigma=0.001, map_size=50, phi=0.5, knn=5, archive_size=5000) :
		super().__init__(env, lamb, algo_name='ns_es', nb_epoch=N, map_size=map_size)
		# optimizer
		self.optimizer=Canonical( mean=np.zeros(shape=ray.get(self.population[0].genome.remote()).shape), sigma=sigma, lamb=lamb)
		# params
		self.lamb=lamb
		self.phi=phi
		# archive
		self.archive=[]
		self.archive_size=archive_size
		self.knn=knn
		# update name


	
	def process(self, epoch):
		# ask genome 
		genome=[self.optimizer.ask() for i in range(self.lamb)]
		# set genome
		ray.get([ i.set_genome.remote(g) for (i,g) in zip(self.population,genome)])
		# add buffer
		for r in self.archive : self.archive.append(r['bcoord'])
		self.archive=self.archive[len(self.archive)-self.archive_size:] if self.archive_size<len(self.archive) else self.archive
		# eval
		res=ray.get([i.eval.remote(self.env) for i in self.population])
		# add archive
		for r in self.archive : self.archive.append(r['bcoord'])
		# fis
		fis=list(map(self.get_novelty,res))
		# fes
		fes=list(map(lambda r : r['value'], res))
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		mu_fis, std_fis= np.mean(fis), np.std(fis) #fis
		min_fes, max_fes= np.min(fes), np.max(fes) #fes
		min_fis, max_fis= np.min(fis), np.max(fis) #fis
		max_min_fes=max_fes-min_fes+self.normalize
		max_min_fis=max_fis-min_fis+self.normalize
		# solutions
		sol=list(map(lambda fe, fi, r : (np.array(r['genome']),-1*(self.phi*(fe-min_fes)/max_min_fes+(1-self.phi)*(fi-min_fis)/max_min_fis ) ) , fes, fis, res))
		# optimizer step 
		self.optimizer.tell(sol)
		# data
		data={
			self.prefix+'/'+'fe_max' : np.max(fes),
			self.prefix+'/'+'fe_mean' : mu_fes,
			self.prefix+'/'+'fe_std' : std_fes,
			self.prefix+'/'+'ns_es'+'/'+'fi_mean' : mu_fis,
			self.prefix+'/'+'ns_es'+'/'+'fi_max' : np.max(fis),
			self.prefix+'/'+'ns_es'+'/'+'fi_std' : std_fis,
		}
		# pop_coord
		pop_coord=list(map(lambda r : r['bcoord'], res))
		# pop_genome
		pop_genome=list(map(lambda r : list(r['genome']), list(filter(lambda r : r['value'] > 1 ,res))))
		return data, pop_coord, pop_genome


	
	def get_novelty(self, r):
		b = r['bcoord']
		# # add buffer
		# self.archive.append(b)
		# self.archive=self.archive[len(self.archive)-self.archive_size:] if self.archive_size<len(self.archive) else self.archive
		if len(self.archive)<=1:
			return 0.0
		# novelty score
		b_A=np.array(self.archive)
		# print('b_A : ',b_A.shape)
		b_D=np.repeat(np.expand_dims(b,axis=0), len(self.archive), axis=0)
		# print('b_D : ', b_D.shape)
		b_S=np.sqrt(np.sum(np.square(b_A-b_D),axis=1))
		b_S.sort()
		# print('b_S : ',b_S.shape)
		fi=np.sum(b_S[:self.knn])
		return fi
	


if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Stacker()
	# env=Ball_in_cup()
	# env=Finger()
	env=CMaze(filename='SNAKE')
	algo=NS_es(env, lamb=28, N=2000, sigma=0.002, map_size=50, phi=0.5, knn=20, archive_size=5000)
	algo.train()