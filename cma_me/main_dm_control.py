from env.dm_control.Ball_in_cup import Ball_in_cup
from env.dm_control.Stacker import Stacker
from env.dm_control.Finger import Finger
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
	# Ball_in_cup
	env=Ball_in_cup(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=CMA_me( env=env, lamb=56, N=1000, sigma=0.001,map_size=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Finger
	env=Finger(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=CMA_me( env=env, lamb=56, N=2000, sigma=0.001,map_size=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Stacker
	env=Stacker(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=CMA_me( env=env, lamb=56, N=2000, sigma=0.001,map_size=50)
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


