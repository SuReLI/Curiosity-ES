from env.GymMaze.CMaze import CMaze
from curiosity_es import Curiosity_es
import wandb
import ray
import os

# check node 
try :
	slurm=os.environ['XPSLURM']
except:
	slurm='false'

for k in range(10):
	# SNAKE
	env=CMaze(filename='SNAKE', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Curiosity_es(env=env, lamb=28, N=2000, sigma=0.05, alpha_icm=1e-4, p=64, phi=0.8, gamma=0.99, beta=0.01, m=50, map_size=50, batch_size=64, max_buffer=int(1e6), cm=1, std_max=None, n_most=0)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# US
	env=CMaze(filename='US', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Curiosity_es(env=env, lamb=28, N=2000, sigma=0.05, alpha_icm=1e-4, p=64, phi=0.8, gamma=0.99, beta=0.01, m=50, map_size=50, batch_size=64, max_buffer=int(1e6), cm=1, std_max=None, n_most=0)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# HARD
	env=CMaze(filename='HARD3', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Curiosity_es(env=env, lamb=28, N=2000, sigma=0.1, alpha_icm=1e-4, p=64, phi=0.8, gamma=0.99, beta=0.01, m=50, map_size=50, batch_size=64, max_buffer=int(1e6), cm=1, std_max=None, n_most=0)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


