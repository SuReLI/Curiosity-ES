from env.dm_control.Ball_in_cup import Ball_in_cup
from env.dm_control.Stacker import Stacker
from env.dm_control.Finger import Finger
from curiosity_GA import Curiosity_GA
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
	algo=Curiosity_GA( lambd=56, mu=7,  N=2000, sigma=0.1, map_size=50 , beta=0.9,alpha_icm=1e-4, p=32, phi=0.8, gamma=0.99, m=50, batch_size=128, max_buffer=int(5*1e5) )
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# Finger
	env=Finger(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Curiosity_GA( lambd=56, mu=7,  N=2000, sigma=0.1, map_size=50 , beta=0.9,alpha_icm=1e-4, p=32, phi=0.8, gamma=0.99, m=50, batch_size=128, max_buffer=int(5*1e5) )
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()

	# # Stacker
	env=Stacker(time_horizon=1000)
	# ray init
	ray.init() if slurm=='false' else ray.init(address='auto')
	algo=Curiosity_GA( lambd=56, mu=7,  N=2000, sigma=0.1, map_size=50 , beta=0.9,alpha_icm=1e-4, p=32, phi=0.8, gamma=0.99, m=50, batch_size=128, max_buffer=int(5*1e5) )
	algo.train()
	# shutdown
	ray.shutdown()
	wandb.finish()


