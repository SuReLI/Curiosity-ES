from utils.algo import Algorithm
from utils.individual import Individual
from utils.canonical import Canonical
from utils.ae import AE
import ray
import numpy as np


class Aurora(Algorithm):

	def __init__(self, env, lamb=56, N=2000, sigma=0.001,map_size=50, alpha=1e-4, m=40, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10, beta= 0.2, knn=20, n_updae_ae=50) :
		super().__init__(env, lamb, algo_name='aurora', nb_epoch=N, map_size=map_size)
		# archive
		self.archive=[]
		self.size_archive=size_archive
		# autoencoder
		self.ae=AE(env, alpha, m, max_buffer, cross_eval, min_average_evolution, ema_beta_d)
		# genome shape
		self.genome_shape=ray.get(self.population[0].genome.remote()).shape
		# params
		self.lamb=lamb
		self.std=sigma
		self.beta=beta #QD tradeoff
		self.knn=knn
		self.n_update_ae=n_updae_ae
		# update name


	
	def process(self, epoch):
		# SELECTION
		best_gen=self.selection()
		# MUTATION
		genome=self.mutation(best_gen)
		# set genome
		ray.get([ i.set_genome.remote(g) for (i,g) in zip(self.population,genome)])
		# eval
		res=ray.get([i.eval.remote(self.env) for i in self.population])
		# fes
		fes=list(map(lambda r : r['value'], res))
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		# add archive
		for r in res : self.add_archive(r)
		# recompute novelty with the updated archive 
		for a in self.archive: a['fi']=self.get_novelty(a['BDR'])
		# update archive
		self.update_archive()
		# update AE then recompute behaviour
		if epoch%self.n_update_ae==0:
			self.ae.update() #ae uptdate with cross val score
			for a in self.archive: a['BDR']=self.ae.encoder(a['BNDR'])
			for a in self.archive: a['fi']=self.get_novelty(a['BDR'])
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
		return data, pop_coord, pop_genome

	def mutation(self, genome):
		offspring=[genome+np.random.normal(0, self.std, self.genome_shape) for _ in range(self.lamb)]
		return offspring
	
	def selection(self):
		fes=list(map(lambda r : r['fe'], self.archive))
		fis=list(map(lambda r : r['fi'], self.archive))
		mu_fes, std_fes= np.mean(fes), np.std(fes) 
		mu_fis, std_fis= np.mean(fis), np.std(fis) #fes
		min_fes, max_fes= np.min(fes), np.max(fes) #fes
		min_fis, max_fis= np.min(fis), np.max(fis) #fis
		max_min_fes=max_fes-min_fes+self.normalize
		max_min_fis=max_fis-min_fis+self.normalize
		i_max=max(self.archive, key=lambda r : self.beta*((r['fe']-min_fes)/max_min_fes)+(1-self.beta)*((r['fi']-min_fis)/max_min_fis)) if len(self.archive) >0 else np.zeros(shape=self.genome_shape)
		return i_max
		
	def add_archive(self,r):
		BNDR=self.ae.get_BNDR(r['S'],r['A'])
		BDR=self.ae.get_BDR(np.expand_dims(BNDR,axis=0)) #BDR
		novelty = self.get_novelty(BDR)
		value = r['value']
		score = self.beta*value+(1-self.beta)*novelty  #QD score
		if len(self.archive)<self.size_archive or score>min(self.archive, key= lambda r :  self.beta*r['fe']+(1-self.beta)*r['fi']) :
			sol={'genome': r['genome'],
			'BNDR':BNDR,
			'BDR':BDR,
			'fi':novelty,
			'fe':r['value']}
			self.ae.add_buffer(BNDR) #update AE dataset 

	def update_archive(self):
		self.archive.sort(key=lambda r :  self.beta*r['fe']+(1-self.beta)*r['fi'] )
		delete_ids=len(self.archive)-self.size_archive if self.size_archive<len(self.archive) else 0
		self.archive=self.archive[delete_ids:]
	
	def get_novelty(self, BDR):
		if len(self.archive)<=1:
			return 0.0
		# novelty score
		b_A=np.array([a['BDR'] for a in self.archive])
		# print('b_A : ',b_A.shape)
		b_D=np.repeat(BDR, len(self.archive), axis=0)
		# print('b_D : ', b_D.shape)
		b_S=np.sqrt(np.sum(np.square(b_A-b_D),axis=1))
		b_S.sort()
		fi=np.sum(b_S[:self.knn])
		return fi


	


if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Stacker()
	# env=Ball_in_cup()
	env=CMaze(filename='SNAKE')
	algo=Aurora( env, lamb=56, N=2000, sigma=0.001,map_size=50, alpha=1e-4, m=40, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10, beta= 0.2, knn=5, n_updae_ae=5)
	for k in range(30):
		algo.process(k)
		print(k)
	# algo.train()