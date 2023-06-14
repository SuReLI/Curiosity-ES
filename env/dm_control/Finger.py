from dm_control import suite 
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from gym.spaces import Box
import numpy as np

class Finger:
	def __init__(self, time_horizon=1000):
		self.time_horizon=time_horizon
		self.env=suite.load("finger","turn_hard",task_kwargs={"time_limit":float("inf")})
		self.observation_space_dim=12
		self.action_space_dim=2
		self.b_space=3
		self.action_space=Box(low=np.zeros(self.action_space_dim), high=np.ones(self.action_space_dim))
		self.observation_space=Box(low=-1*np.pi*np.ones(self.observation_space_dim), high=np.pi*np.ones(self.observation_space_dim))
		self.b_space_gym=Box(low=np.array([-np.pi, -np.pi, -0.14, -0.14 ]), high=np.array([np.pi, np.pi, 0.14, 0.14 ]))
		self.time=0
		self.type='dm_control'
		self.name='finger'

	def __reduce__(self):
		deserializer = Finger
		serialized_data = (self.time_horizon,)
		return deserializer, serialized_data

	def reset(self):
		# self.env.reset()
		# qpos 
		self.env._physics.named.data.qpos["proximal"]=np.pi/2
		self.env._physics.named.data.qpos["distal"]=0
		self.env._physics.named.data.qpos["hinge"]=np.pi/2
		# self.env._physics.named.data.qpos["proximal"]=np.pi/2
		# self.env._physics.named.data.qpos["distal"]=0
		# self.env._physics.named.data.qpos["hinge"]=np.pi/2
		# site_pose
		self.env._physics.model.site_pos=np.array([[ 0.3168727, 0., 0.45692778]
													,[ 0.01, 0. , -0.17 ]
													,[-0.01, 0. , -0.17 ]
													,[ 0.  , 0.  , 0.13 ]])
		# site_size
		self.env._physics.model.site_size=np.array([[0.03,  0.005, 0.005]
													,[0.025, 0.03,  0.025]
													,[0.025, 0.03,  0.025]
													,[0.02,  0.005, 0.005]])
		# site_pose
		
		# # site_size
		# self.env._physics.model.site_size=np.array([[0.03,  0.005, 0.005]
		# 											,[0.025, 0.03,  0.025]
		# 											,[0.025, 0.03,  0.025]
		# 											,[0.02,  0.005, 0.005]])
		self.env._task.before_step([0,0], self.env._physics)
		self.env._physics.step(self.env._n_sub_steps)
		self.env._task.after_step(self.env._physics)
		state=self.env._task.get_observation(self.env._physics)
		position,velocity,touch,target_position,dist_to_target=state["position"],state["velocity"],state["touch"],state["target_position"],np.expand_dims(state["dist_to_target"],axis=0)
		state=np.concatenate((position,velocity,touch,target_position,dist_to_target),axis=0)
		self.time=0
		return state

	def step(self,action):
		self.env._task.before_step(action, self.env._physics)
		self.env._physics.step(self.env._n_sub_steps)
		self.env._task.after_step(self.env._physics)
		state=self.env._task.get_observation(self.env._physics)
		reward=self.env._task.get_reward(self.env._physics)
		# transorfmation
		position,velocity,touch,target_position,dist_to_target=state["position"],state["velocity"],state["touch"],state["target_position"],np.expand_dims(state["dist_to_target"],axis=0)
		state=np.concatenate((position,velocity,touch,target_position,dist_to_target),axis=0)
		done= False
		if self.time>=self.time_horizon-1:
			done=True
		self.time+=1
		if reward<0.1: 
			reward=0
		else : 
			reward=1000*(1.0-self.time/self.time_horizon)
			done=True
		reward=0.0
		return state, reward, done, {'bcoord':list(position)}

	def update_map(self,map,bcoord):
		xnorm=(bcoord[0]+np.pi)*(map.shape[0])/(2*np.pi)
		ynorm=(bcoord[1]+np.pi)*(map.shape[1])/(2*np.pi)
		theta_hinge=np.arctan2(bcoord[2],bcoord[3])
		znorm=(bcoord[2]+np.pi)*(map.shape[2])/(2*np.pi)
		# znorm=(bcoord[2]+0.14)*(map.shape[2])/(2*0.14)
		# gnorm=(bcoord[3]+0.14)*(map.shape[3])/(2*0.14)
		map_coord=[np.minimum(round(xnorm),map.shape[0]-1),np.minimum(round(ynorm),map.shape[1]-1),np.minimum(round(znorm),map.shape[2]-1)]
		# update
		map[tuple(map_coord)]=1 if  map[tuple(map_coord)]==0 else True 
		return map

if __name__=='__main__':
	# env=suite.load("finger","turn_hard")
	# print(env.reset())
	# env._physics.named.data.sensordata["proximal"]=0.0
	# print(env._physics.named.data.sensordata["proximal"])
	env=Finger()
	# print(env.observation_space)
	# print('reset : ',env.reset())
	# state, reward, done, info= env.step(np.zeros(env.action_space.shape))
	# print('state : ', state)
	# print('reward : ',reward)
	import matplotlib.pyplot as plt 
	max_frame = 10000

	width = 480
	height = 480
	video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)
	plt.ion()
	fig,ax=plt.subplots()
	s=env.reset()
	for i in range(100):
		state,reward,d,i=env.step(np.array([[-1],[-0.1]]))
		video = np.hstack([env.env.physics.render(height, width, camera_id=0),
								env.env.physics.render(height, width, camera_id=1)])

		img = ax.imshow(video)
		# time.sleep(1)
		# plt.pause(0.01)  # Need min display time > 0.0.
		fig.canvas.draw()
		fig.canvas.flush_events()
	# print(env.env.action_spec())
	# state=env.env._task.get_observation(env.env._physics)
	# print('position : ',state["position"])
	# print('velocity : ',state["velocity"])
	# print('touch : ',state["touch"])
	# print('target_position : ',state["target_position"])
	# print("dist_to_target : ",np.expand_dims(state["dist_to_target"],axis=0))

