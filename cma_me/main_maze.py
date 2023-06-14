from env.GymMaze.CMaze import CMaze
from cma_me import CMA_me
import wandb
import ray
import os

# check node 
try :
	slurm=os.environ['XPSLURM']
except:
	slurm='false'

for k in range(5):
	# SNAKE
	env=CMaze(filename='SNAKE', time_horizon=2000,n_beams=0)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=CMA_me( env=env, lamb=56, N=2000, sigma=0.001,map_size=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# US
	env=CMaze(filename='US', time_horizon=2000,n_beams=0)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=CMA_me( env=env, lamb=56, N=2000, sigma=0.001,map_size=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# HARD
	env=CMaze(filename='HARD3', time_horizon=2000,n_beams=0)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=CMA_me( env=env, lamb=56, N=2000, sigma=0.001,map_size=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


