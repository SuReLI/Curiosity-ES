from utils.algo import Algorithm
from utils.individual import Individual
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer
import ray
import numpy as np


class Map_elites(Algorithm):

	def __init__(self, env, lamb=56, N=2000, sigma=0.001,map_size=50) :
		super().__init__(env, lamb, algo_name='map_elites_new_2', nb_epoch=N, map_size=map_size)
		[print((low,high)) for (low,high) in zip(tuple(env.b_space_gym.low), tuple(env.b_space_gym.high))]
		[print(map_size) for _ in range(env.b_space_gym.shape[0])]
		# archive 
		self.archive= GridArchive(dims=[map_size for _ in range(env.b_space_gym.shape[0])], ranges=[(low,high) for (low,high) in zip(tuple(env.b_space_gym.low), tuple(env.b_space_gym.high))])
		# emitter
		self.emitters=[GaussianEmitter(self.archive, x0=np.zeros(shape=ray.get(self.population[0].genome.remote()).shape), sigma0= sigma, batch_size=lamb )]
		# optimizer
		self.optimizer = Optimizer(self.archive, self.emitters)
		# params
		self.lamb=lamb
		# update name


	
	def process(self, epoch):
		# ask genome 
		genome=self.optimizer.ask()
		# set genome
		ray.get([ i.set_genome.remote(g) for (i,g) in zip(self.population,genome)])
		# eval
		res=ray.get([i.eval.remote(self.env) for i in self.population])
		# fes
		fes=list(map(lambda r : r['value'], res))
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		# objectives
		objectives=fes
		# BC
		bcs=list(map(lambda r : r['bcoord'], res))
		# optimizer step 
		self.optimizer.tell(objectives,bcs)
		# data
		data={
			self.prefix+'/'+'fe_max' : np.max(fes),
			self.prefix+'/'+'fe_mean' : mu_fes,
			self.prefix+'/'+'fe_std' : std_fes,
		}
		print('fe_max : ', np.max(fes))
		# pop_coord
		pop_coord=list(map(lambda r : r['bcoord'], res))
		# pop_genome
		pop_genome=list(map(lambda r : list(r['genome']), list(filter(lambda r : r['value'] > 1 ,res))))
		return data, pop_coord, pop_genome


	
	


if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	env=Finger(time_horizon=1000)
	algo=Map_elites(env, lamb=56, sigma=0.005)
	# algo.process(0)
	algo.train()