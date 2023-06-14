from ray.rllib.algorithms.ddpg.ddpg_tf_policy import DDPGTF2Policy 
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.utils.annotations import override
from typing import Dict, Tuple, List, Type, Union, Optional, Any
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import Episode
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
    postprocess_nstep_and_prio,
    PRIO_WEIGHTS,
)
from icm import ICM

class CustomPolicy(DDPGTF2Policy):
	def __init__(self, observation_space,action_space,config,*,existing_inputs=None,existing_model=None,):
		self.eta=0.2
		self.icm=ICM( observation_space=observation_space,action_space=action_space, m=50)
		super().__init__( observation_space=observation_space,action_space=action_space,config=config,existing_inputs=existing_inputs,existing_model=existing_model)
		
		
	@override(EagerTFPolicyV2)
	def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[Any, SampleBatch]] = None, episode: Optional[Episode] = None,):
		states=sample_batch[SampleBatch.CUR_OBS]
		actions=sample_batch[SampleBatch.ACTIONS]
		# add curiosity bonus 
		intrinsic_reward=self.icm.get_curiosity(states, actions)
		rewards=sample_batch[SampleBatch.REWARDS]
		sample_batch[SampleBatch.REWARDS]=rewards+self.eta*intrinsic_reward
		# add icm buffer
		self.icm.add_buffer(states, actions)
		# update icm 
		self.icm.update()
		return postprocess_nstep_and_prio(self, sample_batch, other_agent_batches, episode)

	# @override(EagerTFPolicyV2)
	# def learn_on_batch(self, postprocessed_batch):
	# 	stats=super().learn_on_batch(postprocessed_batch)
	# 	self.icm.update()
	# 	return stats


	def change_start(self):
		self.icm.start+=1


