from dm_control import suite 
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from gym.spaces import Box
import numpy as np
from dm_control.suite.stacker import _TIME_LIMIT, Physics, make_model, Stack, _CONTROL_TIMESTEP


def stack_1(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns stacker task with 2 boxes."""
  n_boxes = 1
  physics = Physics.from_xml_string(*make_model(n_boxes=n_boxes))
  task = Stack(n_boxes=n_boxes,
               fully_observable=fully_observable,
               random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)

class Stacker:
	def __init__(self, time_horizon=1000):
		self.time_horizon=time_horizon
		# self.env=suite.load("stacker","stack_2",task_kwargs={"time_limit":float("inf")})
		self.env= stack_1(fully_observable=True, time_limit=float("inf"))
		self.observation_space_dim=42
		self.action_space_dim=5
		self.b_space=2 #3
		self.action_space=Box(low=np.zeros(self.action_space_dim), high=np.ones(self.action_space_dim))
		self.observation_space=Box(low=-1*np.ones(self.observation_space_dim), high=1*np.ones(self.observation_space_dim))
		self.b_space_gym=Box(low=-1*np.ones(self.b_space), high=np.ones(self.b_space))
		self.time=0
		self.type='dm_control'
		self.name='stacker'

		

	def reset(self):
		# self.env.reset()
		# set arm position 
		self.env._physics.named.data.qpos["arm_root"]=np.pi/2
		self.env._physics.named.data.qpos["arm_shoulder"]=0.0
		self.env._physics.named.data.qpos["arm_elbow"]=0.0
		self.env._physics.named.data.qpos["arm_wrist"]=0.0
		self.env._physics.named.data.qpos["thumb"]=0.0
		self.env._physics.named.data.qpos["thumbtip"]=0.0
		self.env._physics.named.data.qpos["finger"]=0.0
		self.env._physics.named.data.qpos["fingertip"]=0.0
		# boxes
		self.env._physics.named.data.qpos["box0_x"]=-0.3
		self.env._physics.named.data.qpos["box0_y"]=0.0
		self.env._physics.named.data.qpos["box0_z"]=0.04
		# self.env._physics.named.data.qpos["box1_x"]=0.3
		# self.env._physics.named.data.qpos["box1_y"]=0.0
		# self.env._physics.named.data.qpos["box1_z"]=0.04
		# target
		self.env._physics.named.model.body_pos["target","x"]=0.0
		self.env._physics.named.model.body_pos["target","y"]=0.0
		self.env._physics.named.model.body_pos["target","z"]=0.07
		state=self.env._task.get_observation(self.env._physics)
		arm_pos,arm_vel,touch,hand_pos,box_pos,box_vel,target_pos=state["arm_pos"].flatten(),state["arm_vel"].flatten(),state["touch"].flatten(),state["hand_pos"].flatten(),state["box_pos"].flatten(),state["box_vel"].flatten(),state["target_pos"].flatten()
		state=np.concatenate((arm_pos,arm_vel,touch,hand_pos,box_pos,box_vel,target_pos),axis=0)
		self.time=0
		return state

	def __reduce__(self):
		deserializer = Stacker
		serialized_data = (self.time_horizon,)
		return deserializer, serialized_data

	def step(self,action):
		# step=self.env.step(action)
		# state,reward=step.observation,step.reward
		self.env._task.before_step(action, self.env._physics)
		self.env._physics.step(self.env._n_sub_steps)
		self.env._task.after_step(self.env._physics)
		state=self.env._task.get_observation(self.env._physics)
		reward=self.env._task.get_reward(self.env._physics)
		# transformation
		arm_pos,arm_vel,touch,hand_pos,box_pos,box_vel,target_pos=state["arm_pos"].flatten(),state["arm_vel"].flatten(),state["touch"].flatten(),state["hand_pos"].flatten(),state["box_pos"].flatten(),state["box_vel"].flatten(),state["target_pos"].flatten()
		state=np.concatenate((arm_pos,arm_vel,touch,hand_pos,box_pos,box_vel,target_pos),axis=0)
		position=list(hand_pos[:2])
		# position.pop(1)
		# box_1=list(box_pos[:2])
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
		return state, reward, done, {'bcoord':position}

	def update_map(self, map, bcoord):
		xnorm=(bcoord[0]+1)*map.shape[0]/2.0
		ynorm=(bcoord[1]+1)*map.shape[1]/2.0
		# znorm=(bcoord[2]+1)*map.shape[2]/2.0
		# b_xnorm=(bcoord[0]+1)*map.shape[2]/2.0
		# b_ynorm=(bcoord[1]+1)*map.shape[3]/2.0
		# b_znorm=(bcoord[2]+1)*map.shape[5]/2.0
		# map_coord=[np.minimum(round(xnorm),map.shape[1]-1),np.minimum(round(ynorm),map.shape[1]-1),np.minimum(round(znorm),map.shape[1]-1), np.minimum(round(b_xnorm),map.shape[1]-1), np.minimum(round(b_ynorm),map.shape[1]-1), np.minimum(round(b_znorm),map.shape[1]-1)]
		map_coord=[np.minimum(round(xnorm),map.shape[1]-1),np.minimum(round(ynorm),map.shape[1]-1)]

		# update
		map[tuple(map_coord)]=1 if  map[tuple(map_coord)]==0 else True 
		return map

if __name__=='__main__':
	env=Stacker()
	s=env.reset()
	print(s.shape)
	for k in range(10):
		state,reward,d,i=env.step(np.array([[1],[-0.5],[0],[0],[0]]))
	# print(env.observation_space)
	# print('reset : ',env.reset())
	# state, reward, done, info= env.step([0 for k in range(5)])
	# print('state : ', state)
	# print('reward : ',reward)
	# print(env.env.action_spec())
	# print(env.env.observation_spec())
	# print(env.reset().shape)

	# obs=env.env._task.get_observation(env.env._physics)
	# print("arm_pos : ",obs["arm_pos"])
	# print("arm_vel : ",obs["arm_vel"])
	# print("touch : ",obs["touch"])
	# print("hand_pos : ",obs["hand_pos"])
	# print("box_pos : ",obs["box_pos"])
	# print("box_vel : ",obs["box_vel"])
	# print("target_pos : ",obs["target_pos"])
	# import matplotlib.pyplot as plt 
	# max_frame = 10000

	# width = 480
	# height = 480
	# video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)
	# plt.ion()
	# fig,ax=plt.subplots()
	# s=env.reset()
	# for i in range(100):
	# 	state,reward,d,i=env.step(np.array([[1],[-0.5],[0],[0],[0]]))
	# 	video = np.hstack([env.env.physics.render(height, width, camera_id=0),
	# 							env.env.physics.render(height, width, camera_id=1)])

	# 	img = ax.imshow(video)
		# time.sleep(1)
		# plt.pause(0.01)  # Need min display time > 0.0.
		# fig.canvas.draw()
		# fig.canvas.flush_events()
	# print(env.env.action_spec())

