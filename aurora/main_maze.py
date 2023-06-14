from env.GymMaze.CMaze import CMaze
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
	env=CMaze(filename='SNAKE', time_horizon=2000,n_beams=8)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Aurora( env, lamb=56, N=1000, sigma=0.001,map_size=50, alpha=1e-4, m=10, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10000, beta= 0.2, knn=10, n_updae_ae=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Finger
	env=CMaze(filename='US', time_horizon=2000,n_beams=8)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Aurora( env, lamb=56, N=1000, sigma=0.001,map_size=50, alpha=1e-4, m=10, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10000, beta= 0.2, knn=10, n_updae_ae=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Stacker
	env=CMaze(filename='HARD', time_horizon=2000,n_beams=16)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Aurora( env, lamb=56, N=1000, sigma=0.001,map_size=50, alpha=1e-4, m=10, max_buffer=50000, cross_eval=5, min_average_evolution=20, ema_beta_d=0.8, size_archive=10000, beta= 0.2, knn=10, n_updae_ae=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


