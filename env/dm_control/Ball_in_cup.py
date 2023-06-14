from dm_control import suite 
from gym.spaces import Box
import numpy as np


# for domain_name, task_name in suite.BENCHMARKING:
	# print("domain : "+domain_name+"  task : "+task_name)

class Ball_in_cup:
	def __init__(self, time_horizon=1000):
		self.action_space_dim=2
		self.observation_space_dim=8
		self.time_horizon=time_horizon
		self.env=suite.load("ball_in_cup","catch", task_kwargs={"time_limit":float("inf")})
		self.x_init=0.0
		self.z_init=-0.05
		self.b_space=4
		self.action_space=Box(low=np.zeros(self.action_space_dim), high=np.ones(self.action_space_dim))
		self.observation_space=Box(low=-1*np.ones(self.observation_space_dim), high=np.ones(self.observation_space_dim))
		self.b_space_gym=Box(low=np.array([-0.5, -0.5, -1, -1]), high=np.array([0.5, 0.5, 1, 1]))
		self.time=0
		self.type='dm_control'
		self.name='ball_in_cup'

	def __reduce__(self):
		deserializer = Ball_in_cup
		serialized_data = (self.time_horizon,)
		return deserializer, serialized_data
		
	def reset(self):
		self.env.reset()
		self.env._physics.named.data.qpos['ball_x']=self.x_init
		self.env._physics.named.data.qpos['ball_z']=self.z_init
		state=self.env._task.get_observation(self.env._physics)
		position,velocity=state["position"],state["velocity"]
		state=np.concatenate((position,velocity), axis=0)
		self.time=0
		return state

	def step(self,action):
		step=self.env.step(action)
		state=step.observation
		reward=step.reward
		# transformation
		position,velocity=state["position"],state["velocity"]
		state=np.concatenate((position,velocity), axis=0)
		done= False
		if self.time>=self.time_horizon-1:
			done=True
		self.time+=1
		if reward==None or reward<1: 
			reward=0
		else : 
			reward=1000*(1.0-self.time/self.time_horizon)
			done=True
		return state, reward, done, {'bcoord':list(position)}

	def update_map(self,map,bcoord):
		xnorm=(bcoord[0]+0.5)*(map.shape[0])
		ynorm=(bcoord[1]+0.5)*(map.shape[1])
		znorm=(bcoord[2]+1)*(map.shape[2])/2.0
		wnorm=(bcoord[3]+1)*(map.shape[3])/2.0
		map_coord=[np.minimum(round(xnorm),map.shape[1]-1),np.minimum(round(ynorm),map.shape[1]-1),np.minimum(round(znorm),map.shape[1]-1),np.minimum(round(wnorm),map.shape[1]-1)]
		# update
		map[tuple(map_coord)]=1 if  map[tuple(map_coord)]==0 else True 
		return map



if __name__=='__main__':
	env=Ball_in_cup()
	# print(env.observation_space)
	# print('reset : ',env.reset())
	# done=False
	# while not done : state, reward, done, info= env.step(env.action_space.sample())
	# print('state : ', state)
	# print('reward : ',reward)
	print(env.env.action_spec())
	# state=env.env._task.get_observation(env.env._physics)
	# print('position : ',state["position"])
	# print('velocity : ',state["velocity"])


