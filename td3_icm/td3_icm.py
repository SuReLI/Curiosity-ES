# import logging
# from typing import List, Optional, Type, Union

# from ray.rllib.algorithms.algorithm import Algorithm
# from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
# from ray.rllib.algorithms.simple_q.simple_q_tf_policy import (
#     SimpleQTF1Policy,
#     SimpleQTF2Policy,
# )
# from ray.rllib.algorithms.simple_q.simple_q_torch_policy import SimpleQTorchPolicy
# from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
# from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
# from ray.rllib.policy.policy import Policy
# from ray.rllib.utils import deep_update
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils.deprecation import DEPRECATED_VALUE, Deprecated
# from ray.rllib.utils.metrics import (
#     LAST_TARGET_UPDATE_TS,
#     NUM_AGENT_STEPS_SAMPLED,
#     NUM_ENV_STEPS_SAMPLED,
#     NUM_TARGET_UPDATES,
#     SYNCH_WORKER_WEIGHTS_TIMER,
#     TARGET_NET_UPDATE_TIMER,
# )
# from ray.rllib.utils.replay_buffers.utils import (
#     update_priorities_in_replay_buffer,
#     validate_buffer_config,
# )
# from ray.rllib.utils.typing import ResultDict
from utils.algo import Algorithm
from utils.individual import Individual
from utils.customcallbacks import CustomCallbacks
from env.dm_control.Ball_in_cup import Ball_in_cup
from env.dm_control.Stacker import Stacker
from env.dm_control.Finger import Finger
from env.GymMaze.CMaze import CMaze
# from TD3 import CustomTD3
import gym
import ray
import numpy as np
from ray.rllib.algorithms.td3.td3 import TD3
from ray.rllib.utils.annotations import override
from ddpg_tf_policy import CustomPolicy

class CustomTD3(TD3):
	def __init__(self, config):
		super(TD3, self).__init__(config)

	@override(TD3)
	def get_default_policy_class(self, config):
		return CustomPolicy

class TD3_icm(Algorithm):

	def __init__(self,env_obj,env_config, lambd=56,N=1000, map_size=50, batch_size=2048) :
		env=env_obj(env_config)
		self.lambd=lambd
		seed=np.random.randint(100000)
		self.config={
					"env": env_obj,
					"env_config":env_config,
					'disable_env_checking':True,
					'num_gpus' : 0,
					'num_cpus_per_worker' : 1,
					'num_gpus_per_worker' : 0,
					'log_level': 'ERROR',
					'use_state_preprocessor':True,
					'actor_hiddens': [32, 32],
					'critic_hiddens': [32, 32],
					'log_level':'ERROR',
					'log_to_driver':False,
					# 'model':
					# 		{'custom_model': DDPGTFModel,
					# 		},
					'num_envs_per_worker' : 1,
					'rollout_fragment_length' : env.env_old.time_horizon,
					# 'batch_mode' : 'truncate_episodes',
					# 'horizon' : None,
					# 'preprocessor_pref' : deepmind,
					'gamma' : 0.99,
					'lr' : 0.0005,
					'train_batch_size' : 128,
					'vf_share_layers' : False,
					'exploration_config':{
								"type": "Curiosity",  # <- Use the Curiosity module for exploring.
								"eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
								"lr": 0.00006,  # Learning rate of the curiosity (ICM) module.
								"feature_dim": 32,  # Dimensionality of the generated feature vectors.
								# Setup of the feature net (used to encode observations into feature (latent) vectors).
								"feature_net_config": {
									"fcnet_hiddens": [64,64],
									"fcnet_activation": "relu",
								},
								"inverse_net_hiddens": [32,16],  # Hidden layers of the "inverse" model.
								"inverse_net_activation": "linear",  # Activation of the "inverse" model.
								"forward_net_hiddens": [32],  # Hidden layers of the "forward" model.
								"forward_net_activation": "linear",  # Activation of the "forward" model.
								"beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
								# Specify, which exploration sub-type to use (usually, the algo's "default"
								# exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
								"sub_exploration": {
									"type": "StochasticSampling",
								}
							},
					'seed' :int(seed),
					'capacity' : env.env_old.time_horizon*lambd*10,
					'num_steps_sampled_before_learning_starts' : env.env_old.time_horizon,
					'lr_schedule' : None,
					'tau' : 0.005,
					'twin_q' : True,
					# 'callbacks' : CustomCallbacks,
					'custom_eval_function' : None,
					'framework' : 'torch',
					'wdecay':0.2,
					'num_workers' : 0}
		self.algo=TD3(config=self.config)
		super().__init__(env.env_old, lambd=56, algo_name='td3_icm', nb_epoch=N, map_size=map_size, seed=seed, wandb_set_server=False)

	
	def process(self, epoch):
		pop_coord=[]
		values=[]
		for _ in range(self.lambd):
			# rollout + one training operation 
			self.algo.training_step()
			# eval 
			value, b= self.eval()
			print(b)
			print(value)
			values.append(value)
			pop_coord.append(b)
		# data
		data={
			self.prefix+'/'+'fe_max' : np.max(values),
		}
		pop_genome=[]
		return data, pop_coord, pop_genome

	def eval(self):
		policy=self.algo.get_policy()
		d=False
		value=0
		s=self.env.reset()
		while not d:
			a_stoch,_,info=policy.compute_single_action(s)
			a=info['action_dist_inputs']
			s, r, d, i= self.env.step(a)
			# self.env.render('C')
			value+=r
		# bcoord
		b=i['bcoord']
		return value, b

	# def info_icm(self):
	# 	policy=self.algo.get_policy()
	# 	for name, param in policy.model.named_parameters():
	# 		if name=='_curiosity_feature_net._hidden_layers.1._model.0.weight':
	# 			print(param)

def recursif_dict(d,k=0):
	if isinstance(d,dict) :
		for key in d.keys():
			print(k*str('    ')+key) if isinstance(d[key],dict) else print(k*str('   ')+"'"+key+"'"+' : '+str(d[key]))
			recursif_dict(d[key],k+1)


if __name__=='__main__':
	from custom_env import Ball_in_cup
	from custom_env import Stacker
	from custom_env import Finger
	from custom_env import Maze
	import time
	# ray.init(log_to_driver=False)
	td3=TD3_icm(env_obj=Maze,env_config={ 'time_horizon':1000, 'filename': 'SNAKE', 'n_beams':8}, lambd=1, N=1000, map_size=50, batch_size=2048)
	td3.train()
	# td3.td3.train()
	# policy=td3.td3.get_policy()
	# print(policy.model)
	# policy.model.base_model.summary()
	# env=td3.env
	# s=env.reset()
	# d=False
	# while not d:
	# 	a_stoch,_,info=policy.compute_single_action(s)
	# 	a=info['action_dist_inputs']
	# 	print(a)
	# 	s, r, d, i= env.step(a)
	# 	env.render('C')
	# model_out= policy.model.forward(np.expand_dims(s,axis=0),[])
	# print(model_out)
	# print(model_out.get_policy_output())

	
	# ray.rllib.utils.check_env(MazeR)
	
	# minimal_conf={{"env":Maze,
	# 				"env_config":{'filename':'SNAKE', 'time_horizon':1000, 'n_beams':8},
	# 				'disable_env_checking':True,
	# 				'num_gpus' : 0,
	# 				'num_cpus_per_worker' : 1,
	# 				'num_gpus_per_worker' : 0,
	# 				'log_level': 'ERROR',
	# 				'use_state_preprocessor':True,
	# 				'actor_hiddens': [32, 32],
	# 				'critic_hiddens': [32, 32],
	# 				'num_envs_per_worker' : 1,
	# 				'gamma' : 0.99,
	# 				'lr' : 0.0005,
	# 				'rollout_fragment_length' :1000,
	# 				'vf_share_layers' : False,
	# 				'seed' : None,
	# 				'capacity' : 1000000,
	# 				'lr_schedule' : None,
	# 				'tau' : 0.005,
	# 				# 'num_steps_sampled_before_learning_starts' : 256,
	# 				'train_batch_size':256,
	# 				'twin_q' : True,
	# 				'custom_eval_function' : None,
	# 				'framework' : 'tf2',
	# 				'num_workers':1,
	# 				'training_intensity':256}}
	# algo=CustomTD3(config=minimal_conf)
	# for k in range(5):
	# 	print('epoch')
	# 	algo.training_step()
	