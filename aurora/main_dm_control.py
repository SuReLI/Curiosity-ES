from env.dm_control.Ball_in_cup import Ball_in_cup
from env.dm_control.Stacker import Stacker
from env.dm_control.Finger import Finger
from aurora import Aurora
import wandb
import ray
import os

# check node 
try :
	slurm=os.environ['XPSLURM']
except:
	slurm='false'
	
for k in range(10):
	# Ball_in_cup
	env=Ball_in_cup(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Aurora( env, lamb=56, N=1000, sigma=0.001,map_size=50, alpha=1e-4, m=10, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10000, beta= 0.2, knn=10, n_updae_ae=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Finger
	env=Finger(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Aurora( env, lamb=56, N=2000, sigma=0.001,map_size=50, alpha=1e-4, m=10, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10000, beta= 0.2, knn=10, n_updae_ae=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Stacker
	env=Stacker(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Aurora( env, lamb=56, N=2000, sigma=0.001,map_size=50, alpha=1e-4, m=10, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10000, beta= 0.2, knn=10, n_updae_ae=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


