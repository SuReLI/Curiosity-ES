from utils.canonical import Canonical
# from utils.cmaes import CMA as Canonical
# from utils.icm_never_give_up import ICM
from utils.icm_opt import ICM

from utils.algo import Algorithm
from utils.cur_individual import Individual
import random
import ray
import numpy as np


class Curiosity_es(Algorithm):

	def __init__(self, env, lamb=56, N=2000, sigma=0.001, alpha_icm=1e-3, p=2, phi=0.9, gamma=0.9, beta=0.2, m=50, map_size=50, batch_size=2048, max_buffer=10000, std_max=0.5, cm=0.8, n_most=0, p_front=30) :
		super().__init__(env, lamb, algo_name='curiosity_es', nb_epoch=N, map_size=map_size)
		# icm
		self.icm=ICM(env, alpha_icm=alpha_icm, p= p, m=m, beta=beta, gamma = gamma,batch_size=batch_size, max_buffer=max_buffer)
		# optimizer
		self.optimizer=Canonical( mean=np.zeros(shape=ray.get(self.population[0].genome.remote()).shape), sigma=sigma, lamb=lamb+n_most, cm=cm)
		# params
		self.phi=phi
		self.lamb=lamb
		self.std_max=std_max
		self.std=sigma
		self.N=N
		self.normalize=0.00001
		self.n_most=n_most
		self.most_curious=[]
		self.rand=[]
		self.p_front=p_front
		# update name


	
	def process(self, epoch):
		self.optimizer._sigma=self.std+epoch*(self.std_max-self.std)/self.N if self.std_max!=None else self.std
		# ask genome 
		genome=[self.optimizer.ask() for i in range(self.lamb)]
		# set genome
		ray.get([ i.set_genome.remote(g) for (i,g) in zip(self.population,genome)])
		# eval
		res_b=ray.get([i.eval.remote(self.env) for i in self.population])
		# fis
		fis=list(map(self.get_curiosity,res_b)) if epoch!=0 else [0.0 for i in self.population]
		for r, fi in zip(res_b,fis) : r['fi']=fi
		res=res_b+self.most_curious+self.rand if epoch!=0 else res_b+res_b[:self.n_most] #keep track
		# fis
		# fis=list(map(self.get_curiosity,res)) if epoch!=0 else [0.0 for i in self.population]
		# fes
		fes=list(map(lambda r : r['value'], res))
		# trick
		# normalize
		mu_fes, std_fes= np.mean(fes), np.std(fes) #fes
		mu_fis, std_fis= np.mean(fis), np.std(fis) #fis
		min_fes, max_fes= np.min(fes), np.max(fes) #fes
		min_fis, max_fis= np.min(fis), np.max(fis) #fis
		max_min_fes=max_fes-min_fes+self.normalize
		max_min_fis=max_fis-min_fis+self.normalize


		# std_fes+=self.normalize
		# std_fis+=self.normalize

		# solutions
		# sol=list(map(lambda fe, fi, fa, r : (np.array(r['genome']),-1*(self.phi*(fe-mu_fes)/std_fes+(1-self.phi)*(self.phi_reg*(fi-mu_fis)/std_fis+ (1-self.phi_reg)*(fa-mu_fa)/std_fa) ) ) , fes, fis, fas, res)) # optimizer minimize
		sol=[]
		for fe, fi, r in zip(fes, fis, res): r['f_fitness']=-1*(self.phi*(fe-min_fes)/max_min_fes+(1-self.phi)*((fi-min_fis)/max_min_fis)) 
		res_sort=sorted(res, key=lambda r: r['f_fitness'])
		self.most_curious=res_sort[:self.n_most]
		sol=[ (np.array(r['genome']),r['f_fitness']) for r in res_sort]
		# sol=list(map(lambda fe, fi, r : (np.array(r['genome']),0 ), fes, fis, res)) # optimizer minimize
		# optimizer step 
		self.optimizer.tell(sol)
		# curiosity update
		self.add_global_buffer(res_b,epoch)
		global_loss, global_li, global_lf = self.icm.update() if epoch % 1==0 else (0,0,0)
		# data
		data={
			self.prefix+'/'+'fe_max' : np.max(fes),
			self.prefix+'/'+'fe_mean' : mu_fes,
			self.prefix+'/'+'fe_std' : std_fes,
			self.prefix+'/'+'curiosity_es/fi_max' : np.max(fis),
			self.prefix+'/'+'curiosity_es/fi_mean' : mu_fis,
			self.prefix+'/'+'curiosity_es/fi_std' : std_fis,
			# self.prefix+'/'+'curiosity_es/LF' : global_lf,
			# self.prefix+'/'+'curiosity_es/LI' : global_li,
			self.prefix+'/'+'curiosity_es/L' : global_loss,
			# self.prefix+'/'+'curiosity_es/LREC' : global_lrec,
			# self.prefix+'/'+'curiosity_es/LNORM' : global_norm,
		}
		print('fe_max : ', np.max(fes))
		print('global loss : ',global_loss)
		print('global lf : ',global_lf)
		print('global li : ', global_li)
		# print('global lreg : ', global_lreg)
		# print('global norm : ', global_norm)
		# pop_coord
		pop_coord=list(map(lambda r : r['bcoord'], res))
		# pop_genome
		pop_genome=list(map(lambda r : list(r['genome']), list(filter(lambda r : r['value'] > 1 ,res))))
		return data, pop_coord, pop_genome


	
	def get_curiosity(self, r):
		S = r['S']
		A = r['A']
		# fi
		fi=self.icm.get_curiosity(S,A)
		r['fi']=fi
		return fi
	
	def add_global_buffer(self, res,epoch):
		if epoch==0: 
			for r in res : self.icm.add_buffer(r['S'],r['A'])
		res_sort=sorted(res, key=lambda r: r['fi'])
		p=int(len(res_sort)*self.p_front/100)
		for r in res_sort[:-p] : self.icm.add_buffer(r['S'],r['A'])
			

if __name__=='__main__':
	from env.dm_control.Ball_in_cup import Ball_in_cup
	from env.dm_control.Stacker import Stacker
	from env.dm_control.Finger import Finger
	from env.GymMaze.CMaze import CMaze
	import wandb
	# for k in range(20):
	env=CMaze(filename='SNAKE',n_beams=16, time_horizon=2000)
	# env=Finger(time_horizon=1000)
	# ray init
	ray.init()
	algo=Curiosity_es(env=env, lamb=28, N=1000, sigma=0.1, alpha_icm=1e-4, p=128, phi=0.9, gamma=0.99, beta=0.01, m=50, map_size=50, batch_size=64, max_buffer=int(1e5), cm=1, std_max=None, n_most=10)

	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()