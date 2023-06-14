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
# from ray.rllib.algorithms.td3.td3 import TD3
# from ddpg_tf_policy import CustomPolicy
# from ray.rllib.utils.annotations import override
# class CustomTD3(TD3):
# 	def __init__(self, config):
# 		super(TD3, self).__init__(config)

# 	@override(TD3)
# 	def get_default_policy_class(self, config):
# 		return CustomPolicy
# 	@override(TD3)
# 	def training_step(self) -> ResultDict:
# 		"""Simple Q training iteration function.
# 		Simple Q consists of the following steps:
# 		- Sample n MultiAgentBatches from n workers synchronously.
# 		- Store new samples in the replay buffer.
# 		- Sample one training MultiAgentBatch from the replay buffer.
# 		- Learn on the training batch.
# 		- Update the target network every `target_network_update_freq` sample steps.
# 		- Return all collected training metrics for the iteration.
# 		Returns:
# 			The results dict from executing the training iteration.
# 		"""
# 		print('TRAINING')
# 		batch_size = self.config.train_batch_size
# 		local_worker = self.workers.local_worker()

# 		# Sample n MultiAgentBatches from n workers.
# 		new_sample_batches = synchronous_parallel_sample(
# 			worker_set=self.workers, concat=False
# 		)

# 		for batch in new_sample_batches:
# 			# Update sampling step counters.
# 			self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
# 			self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
# 			# Store new samples in the replay buffer
# 			self.local_replay_buffer.add(batch)

# 		global_vars = {
# 			"timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
# 		}
# 		# Update target network every `target_network_update_freq` sample steps.
# 		cur_ts = self._counters[
# 			NUM_AGENT_STEPS_SAMPLED
# 			if self.config.count_steps_by == "agent_steps"
# 			else NUM_ENV_STEPS_SAMPLED
# 		]

# 		if cur_ts > self.config.num_steps_sampled_before_learning_starts:
# 			# Use deprecated replay() to support old replay buffers for now
# 			train_batch = self.local_replay_buffer.sample(batch_size)

# 			# Learn on the training batch.
# 			# Use simple optimizer (only for multi-agent or tf-eager; all other
# 			# cases should use the multi-GPU optimizer, even if only using 1 GPU)
# 			if self.config.get("simple_optimizer") is True:
# 				train_results = train_one_step(self, train_batch)
# 			else:
# 				train_results = multi_gpu_train_one_step(self, train_batch)

# 			#************************************ icm learning ************************************
# 			# for pid in self.workers.local_worker().policy_map.keys():
# 			# 		print(self.workers.local_worker().policy_map[pid].icm.S1.shape)
# 			#************************************ icm learning ************************************

# 			# Update replay buffer priorities.
# 			update_priorities_in_replay_buffer(
# 				self.local_replay_buffer,
# 				self.config,
# 				train_batch,
# 				train_results,
# 			)

# 			last_update = self._counters[LAST_TARGET_UPDATE_TS]
# 			if cur_ts - last_update >= self.config.target_network_update_freq:
# 				with self._timers[TARGET_NET_UPDATE_TIMER]:
# 					to_update = local_worker.get_policies_to_train()
# 					local_worker.foreach_policy_to_train(
# 						lambda p, pid: pid in to_update and p.update_target()
# 					)
# 				self._counters[NUM_TARGET_UPDATES] += 1
# 				self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

# 			# Update weights and global_vars - after learning on the local worker -
# 			# on all remote workers (only those policies that were actually trained).
# 			with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
# 				self.workers.sync_weights(
# 					policies=list(train_results.keys()),
# 					global_vars=global_vars,
# 				)
# 		else:
# 			train_results = {}

# 		# Return all collected metrics for the iteration.
# 		return train_results


if __name__=='__main__':
	# import gym
	# from custom_env import Maze, Stacker, Ball_in_cup, Finger
	# minimal_conf={"env":Maze,
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
	# 				'training_intensity':256}
	# algo=CustomTD3(config=minimal_conf)
	# for k in range(5):
	# 	print('epoch')
	# 	algo.training_step()
		# # print(algo.get_policy().icm.start)
		# algo.get_policy().change_start()
		# algo.training_step()
		# print(algo.get_policy().icm.S1.shape)
	# policy=algo.get_policy()
	# print(policy)
	# policy.model.base_model.summary()
	# print(policy.icm.p)
	# policy.icm.p=3
	# algo.train()
	# policy=algo.get_policy()
	# print(policy.icm.p)

	# env=gym.make('MountainCarContinuous-v0')
	# s=env.reset()
	# d=False
	# while not d:
	# 	a_stoch,_,info=policy.compute_single_action(s)
	# 	a=info['action_dist_inputs']
	# 	print(a)
	# 	s, r, d, i= env.step(a)
	# model_out= policy.model.forward(np.expand_dims(s,axis=0),[])
	# print(model_out)
	# print(model_out.get_policy_output())
	from ray.rllib.algorithms.td3.td3 import TD3
	from custom_env import Maze
	minimal_conf={"env":Maze,
					"env_config":{'filename':'SNAKE', 'time_horizon':1000, 'n_beams':8},
					'exploration_config':{
											"type": "Curiosity",  # <- Use the Curiosity module for exploring.
											"eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
											"lr": 0.001,  # Learning rate of the curiosity (ICM) module.
											"feature_dim": 32,  # Dimensionality of the generated feature vectors.
											# Setup of the feature net (used to encode observations into feature (latent) vectors).
											"feature_net_config": {
												"fcnet_hiddens": [64,64],
												"fcnet_activation": "relu",
											},
											"inverse_net_hiddens": [32,16],  # Hidden layers of the "inverse" model.
											"inverse_net_activation": "linear",  # Activation of the "inverse" model.
											"forward_net_hiddens": [32],  # Hidden layers of the "forward" model.
											"forward_net_activation": "relu",  # Activation of the "forward" model.
											"beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
											# Specify, which exploration sub-type to use (usually, the algo's "default"
											# exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
											"sub_exploration": {
												"type": "StochasticSampling",
											}
										},
					'disable_env_checking':True,
					'num_gpus' : 0,
					'num_cpus_per_worker' : 1,
					'num_gpus_per_worker' : 0,
					'log_level': 'ERROR',
					'use_state_preprocessor':True,
					'actor_hiddens': [32, 32],
					'critic_hiddens': [32, 32],
					'num_envs_per_worker' : 1,
					'gamma' : 0.99,
					'lr' : 0.0005,
					'rollout_fragment_length' :1000,
					'vf_share_layers' : False,
					'seed' : None,
					'capacity' : 1000000,
					'lr_schedule' : None,
					'tau' : 0.005,
					# 'num_steps_sampled_before_learning_starts' : 256,
					'train_batch_size':256,
					'twin_q' : True,
					'custom_eval_function' : None,
					'framework' : 'torch',
					'num_workers':0,
					'training_intensity':256}
	algo=TD3(config=minimal_conf)
	policy=algo.get_policy()
	print(policy.model._curiosity_feature_net)
	# for k in range(3):
	# 	print('train')
	# 	algo.train()
	# 	print('end')
