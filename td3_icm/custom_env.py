import gym 
from env.dm_control.Ball_in_cup import Ball_in_cup as OldBall_in_cup
from env.dm_control.Stacker import Stacker as OldStacker
from env.dm_control.Finger import Finger as OldFinger
from env.GymMaze.CMaze import CMaze


class Maze(gym.Env):
	def __init__(self, env_config):
		super().__init__()
		self.config=env_config
		self.env_old=CMaze(filename=env_config['filename'], time_horizon=env_config['time_horizon'], n_beams=env_config['n_beams'])
		self.action_space=self.env_old.action_space
		self.observation_space=self.env_old.observation_space

	def step(self,action):
		return self.env_old.step(action)
	def reset(self):
		return self.env_old.reset()
	def __reduce__(self):
		deserializer = Maze
		serialized_data = (self.config,)
		return deserializer, serialized_data

class Stacker(gym.Env):
	def __init__(self, env_config):
		super().__init__()
		self.config=env_config
		self.env_old=OldStacker(time_horizon=env_config['time_horizon'])
		self.action_space=self.env_old.action_space
		self.observation_space=self.env_old.observation_space

	def step(self,action):
		return self.env_old.step(action)
	def reset(self):
		return self.env_old.reset()
	def __reduce__(self):
		deserializer = Stacker
		serialized_data = (self.config,)
		return deserializer, serialized_data

class Ball_in_cup(gym.Env):
	def __init__(self, env_config):
		super().__init__()
		self.config=env_config
		self.env_old=OldBall_in_cup(time_horizon=env_config['time_horizon'])
		self.action_space=self.env_old.action_space
		self.observation_space=self.env_old.observation_space

	def step(self,action):
		return self.env_old.step(action)
	def reset(self):
		return self.env_old.reset()
	def __reduce__(self):
		deserializer = Ball_in_cup
		serialized_data = (self.config,)
		return deserializer, serialized_data

class Finger(gym.Env):
	def __init__(self, env_config):
		super().__init__()
		self.config=env_config
		self.env_old=OldFinger(time_horizon=env_config['time_horizon'])
		self.action_space=self.env_old.action_space
		self.observation_space=self.env_old.observation_space

	def step(self,action):
		return self.env_old.step(action)
	def reset(self):
		return self.env_old.reset()
	def __reduce__(self):
		deserializer = Finger
		serialized_data = (self.config,)
		return deserializer, serialized_data