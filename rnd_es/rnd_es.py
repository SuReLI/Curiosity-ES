from utils.canonical import Canonical
from utils.rnd import RND
from utils.algo import Algorithm
from utils.cur_individual import Individual
import ray
import numpy as np


class RND_es(Algorithm):

	def __init__(self, env, lamb=56, N=2000, sigma=0.001, alpha_rnd=1e-3, p=2, phi=0.9, gamma=0.9, beta=0.2, m=50, map_size=50, batch_size=2048, max_buffer=10000, std_max=None, cm=0.8, phi_reg=0.7) :
		super().__init__(env, lamb, algo_name='rnd_es', nb_epoch=N, map_size=map_size)
		# icm
		self.icm=RND(env,alpha_rnd=alpha_rnd, p= p, m=m, batch_size=batch_size, max_buffer=max_buffer, gamma=gamma)
		# optimizer
		self.optimizer=Canonical( mean=np.zeros(shape=ray.get(self.population[0].genome.remote()).shape), sigma=sigma, lamb=lamb, cm=cm)
		# params
		self.phi=phi
		self.lamb=lamb
		self.std_max=std_max
		self.std=sigma
		self.phi_reg=phi_reg
		self.N=N
		self.normalize=0.001
		# update name


	
	def process(self, epoch):
		self.optimizer._sigma=self.std+epoch*(self.std_max-self.std)/self.N if self.std_max!=None else self.std
		# ask genome 
		genome=[self.optimizer.ask() for i in range(self.lamb)]
		# set genome
		ray.get([ i.set_genome.remote(g) for (i,g) in zip(self.population,genome)])
		# eval
		res=ray.get([i.eval.remote(self.env) for i in self.population])
		# fis
		fis=list(map(self.get_curiosity,res))
		# fes
		fes=list(map(lambda r : r['value'], res))
		# fgen
		fgens=list(map(lambda r : np.sum(r['genome']**2), res))
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		mu_fis, std_fis= np.mean(fis), np.std(fis) #fis
		mu_fgen, std_fgen= np.mean(fgens), np.std(fgens) #fgen

		std_fes+=self.normalize
		std_fis+=self.normalize
		std_fgen+=self.normalize

		# solutions
		sol=list(map(lambda fe, fi, fgen, r : (np.array(r['genome']),-1*(self.phi*(fe-mu_fes)/std_fes+(1-self.phi)*(self.phi_reg*(fi-mu_fis)/std_fis+ (1-self.phi_reg)*(fgen-mu_fgen)/std_fgen) ) ) , fes, fis, fgens, res)) # optimizer minimize
		# sol=list(map(lambda fe, fi, r : (np.array(r['genome']),0 ), fes, fis, res)) # optimizer minimize
		# optimizer step 
		self.optimizer.tell(sol)
		# curiosity update
		global_loss= self.icm.update() if epoch%5==0 else 0
		# data
		data={
			self.prefix+'/'+'fe_max' : np.max(fes),
			self.prefix+'/'+'fe_mean' : mu_fes,
			self.prefix+'/'+'fe_std' : std_fes,
			self.prefix+'/'+'curiosity_es/fi_max' : np.max(fis),
			self.prefix+'/'+'curiosity_es/fi_mean' : mu_fis,
			self.prefix+'/'+'curiosity_es/fi_std' : std_fis,
			self.prefix+'/'+'curiosity_es/L' : global_loss,
		}
		print('fe_max : ', np.max(fes))
		print('global loss : ',global_loss)
		# pop_coord
		pop_coord=list(map(lambda r : r['bcoord'], res))
		# pop_genome
		pop_genome=list(map(lambda r : list(r['genome']), list(filter(lambda r : r['value'] > 1 ,res))))
		return data, pop_coord, pop_genome


	
	def get_curiosity(self, r):
		S = r['S']
		# add buffer
		self.icm.add_buffer(S)
		# fi
		fi=self.icm.get_curiosity(S)
		return fi.numpy()
	


if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	# env=Finger(time_horizon=1000)
	# env=Stacker(time_horizon=1000)
	env=CMaze(filename='HARD3',n_beams=8, time_horizon=2000)

	# ray init
	ray.init()
	algo=RND_es(env=env, lamb=56, N=2000, sigma=0.5, alpha_rnd=1e-4, p=128, phi=0.8, gamma=0.95, beta=0.2, m=10, map_size=50, batch_size=64, max_buffer=int(5*1e5), phi_reg=0.999, cm=1)
	algo.train()