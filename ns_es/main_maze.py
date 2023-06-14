from env.GymMaze.CMaze import CMaze
from ns_es import NS_es
import wandb
import ray
import os

# check node 
try :
	slurm=os.environ['XPSLURM']
except:
	slurm='false'

for k in range(1):
	# SNAKE
	env=CMaze(filename='SNAKE', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=NS_es(env=env, lamb=56, N=2000, sigma=0.5, map_size=50, phi=0.5, knn=10, archive_size=500000)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# US
	env=CMaze(filename='US', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=NS_es(env=env, lamb=56, N=2000, sigma=0.5, map_size=50, phi=0.5, knn=10, archive_size=500000)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# HARD
	env=CMaze(filename='HARD3', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=NS_es(env=env, lamb=56, N=2000, sigma=0.5, map_size=50, phi=0.5, knn=10, archive_size=500000)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


