from env.GymMaze.CMaze import CMaze
from nslc import Nslc
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
	env=CMaze(filename='SNAKE', time_horizon=2000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Nslc( env, lambd=56, mu=7,  N=2000, sigma=0.5, map_size=50 , beta=0.9, knn=10)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# # US
	# env=CMaze(filename='US', time_horizon=2000)
	# # ray init
	# ray.init() if slurm=='false' else ray.init(address='auto')
	# algo=Nslc( env, lambd=56, mu=7,  N=2000, sigma=0.5, map_size=50 , beta=0.9, knn=5)
	# algo.train()
	# # shutdown
	# ray.shutdown()
	# wandb.finish()

	# # HARD
	# env=CMaze(filename='HARD3', time_horizon=2000)
	# # ray init
	# ray.init() if slurm=='false' else ray.init(address='auto')
	# algo=Nslc( env, lambd=56, mu=7,  N=2000, sigma=0.5, map_size=50 , beta=0.9, knn=5)
	# algo.train()
	# # shutdown
	# ray.shutdown()
	# wandb.finish()


