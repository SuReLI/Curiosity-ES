from env.GymMaze.build.Game import Maze
import os
import numpy as np 
import json
import subprocess
from gym import spaces
import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import ray
import time

class CMaze(Maze):

	def __init__(self, xinit=5, yinit=5, xgoal=90, ygoal=90,width=100,height=100, time_horizon=3000, filename=None, n_beams=8):
		super().__init__(xinit, yinit, xgoal , ygoal, time_horizon, n_beams)
		self.width=width
		self.height=height
		# pygame plot:
		self.screen=None
		self.clock=None
		self.screen_width = 600
		self.screen_height = 600
		self.size_cv=84
		self.n_beams=n_beams
		self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
		self.type='maze'
		self.name=filename if filename!=None else None


		# # matplolib
		# self.figure=plt.figure()

		# gym
		gym.logger.set_level(40)
		self.action_space = spaces.Box(np.array([-5,-5]),np.array([5,5]), dtype=np.float32)
		self.observation_space = spaces.Box(-np.ones(shape=(1,7+self.agent.n_beams)).flatten(), np.ones(shape=(1,7+self.agent.n_beams)).flatten(), dtype=np.float32)
		self.b_space=2
		self.b_space_gym=spaces.Box(low=np.array([0,0,-self.vmax,-self.vmax]), high=np.array([self.width,self.height, self.vmax, self.vmax]))


		# load
		if filename!=None:
			self.filename=filename
			self.load(filename)


		# init block for open cv state 
		self.block_cv=self.init_block()
		self.open_cv_dim=84


	def __reduce__(self):
		deserializer = CMaze
		serialized_data = (self.xinit, self.yinit, self.xgoal, self.ygoal, self.width, self.height, self.time_horizon, self.filename, self.n_beams,)
		return deserializer, serialized_data

	def step(self,action):
		# Cstep
		state, reward, done, info = self.Cstep(action)
		info={'bcoord':self.bcoord()}
		# reward=reward*1000*(1-self.time/self.time_horizon)
		return np.array(state), reward, done, info

	def step_cv(self,action):
		# Cstep
		state, reward, done, info = self.Cstep(action)
		state=self.state_open_cv()
		return state, reward, done, info

	def reset(self):
		# Creset
		state=self.Creset()
		return np.array(state)

	def reset_cv(self):
		# Creset
		state=self.Creset()
		# open_cv
		state=self.state_open_cv()
		return state

	def bcoord(self):
		return [float(self.agent.x),float(self.agent.y),float(self.agent.xdot),float(self.agent.ydot)]

	def update_map(self, map, bcoord):
		xnorm=bcoord[0]*(map.shape[1])/self.width
		ynorm=bcoord[1]*(map.shape[0])/self.height
		map_coord=[np.maximum(-(1+round(ynorm)),-map.shape[0]),np.minimum(round(xnorm),map.shape[1]-1)]
		# update
		map[tuple(map_coord)]=1 if  map[tuple(map_coord)]==0 else True 
		return map

	def render(self, type="py", freeze=False, behaviours=[], curiosities=[], times=[], save_image=False, filename="behaviours", shade=False):
		if type=="py":
			import pygame 
			if self.screen is None or len(behaviours)>0:
				pygame.init()
				pygame.display.init()
				self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
			if self.clock is None:
				self.clock = pygame.time.Clock()

			# Screen
			self.surf = pygame.Surface((self.screen_width, self.screen_height))
			self.surf.fill((255, 255, 255))
			# ratio
			rwidth=self.screen_width/self.width
			rheight=self.screen_height/self.height
			rnorm=np.sqrt(rwidth**2+rheight**2)

			# blocks 
			for block in self.block_list:
				pygame.draw.rect(self.surf, (0,0,0), (block.x*rwidth,block.y*rheight,block.w*rwidth,block.h*rheight), 0) 
			
			# goal
			pygame.draw.circle(self.surf, (0,255,0),(self.xgoal*rwidth, self.ygoal*rheight), self.treshold*rnorm , 0) 

			# start
			
			pygame.draw.circle(self.surf, (0,0,0),(self.xinit*rwidth, self.yinit*rheight), self.agent.r*rnorm , 0) 
			
			treshold=0.9

			if len(behaviours)==0:
				# agent
				pygame.draw.circle(self.surf, (0,0,255),(self.agent.x*rwidth, self.agent.y*rheight), self.agent.r*rnorm , 0) 

				
				# lidar 
				for theta,dist in zip(self.agent.beams,self.agent.lidar_observation()):
					x0,y0=self.agent.x*rwidth,self.agent.y*rheight
					x1,y1=(self.agent.x+dist*np.cos(theta))*rwidth,(self.agent.y+dist*np.sin(theta))*rheight
					pygame.draw.line(self.surf, (255,0,0), (x0, y0), (x1, y1))
				

			elif len(behaviours)>0 and len(curiosities)==0 and len(times)==0:
				for behaviour in behaviours:
					(xscreen, yscreen), rscreen=(behaviour[0]*rwidth,behaviour[1]*rheight),self.agent.r*rnorm
					pygame.draw.circle(self.surf, (0,0, 255),(xscreen, yscreen), rscreen/2, 0)

			elif len(behaviours)>0 and len(curiosities)>0:
				for behaviour, curiosity in zip(behaviours,curiosities):
					(xscreen, yscreen), rscreen=(behaviour[0]*rwidth,behaviour[1]*rheight),self.agent.r*rnorm
					rcolor=(curiosity-np.min(curiosities))/(np.max(curiosities)-np.min(curiosities))
					if not shade:
						rcolor=1
					pygame.draw.circle(self.surf, (255,255*(1-rcolor),255*(1-rcolor)),(xscreen, yscreen), rscreen/2, 0)
				
			elif len(behaviours)>0 and len(times)>0:
				for behaviour, time in zip(behaviours,times):
					(xscreen, yscreen), rscreen=(behaviour[0]*rwidth,behaviour[1]*rheight),self.agent.r*rnorm
					rcolor=(time-np.min(times))/(np.max(times)-np.min(times))
					if not shade:
						rcolor=1
					pygame.draw.circle(self.surf, (255*(1-rcolor),255*(1-rcolor),255),(xscreen, yscreen), rscreen/2, 0)


			# Project surface on screen 
			self.surf = pygame.transform.flip(self.surf, False, True)
			self.screen.blit(self.surf, (0, 0))
			if save_image:
				filename = filename+'.JPG'
				pygame.image.save(self.screen, filename)
				
			if freeze:
				try:
					while 1:
						event = pygame.event.wait()
						if event.type == pygame.QUIT:
							break
						if event.type == pygame.KEYDOWN:
							if event.key == pygame.K_ESCAPE or event.unicode == 'q':
								break
						pygame.display.flip()
				finally:
					pygame.quit() 
			elif not freeze :
				pygame.event.pump()
				self.clock.tick(self.metadata["render_fps"])
				pygame.display.flip()
		else:
			self.Crender()
	
	def render_matplot(self,ax = None, behaviours=[], curiosities=[]):
		# remove patches
		if len(ax.patches)==0:
			# blocks 
			for block in self.block_list:
				rect = matplotlib.patches.Rectangle((block.x, block.y),block.w, block.h,color ='black')
				ax.add_patch(rect)

			# goal
			(x, y), r=(self.xgoal,self.ygoal),self.treshold
			circle=matplotlib.patches.Circle((x, y), radius=r, color="black")
			ax.add_patch(circle)

		# behaviours
		for behaviour in behaviours:
			(x, y), r=(behaviour[0],behaviour[1]),self.agent.r
			circle=matplotlib.patches.Circle((x, y), radius=r/2, color="red")
			ax.add_patch(circle)
			
		# axlimit
		ax.axis(xmin=0,xmax=self.width)
		ax.axis(ymin=0,ymax=self.height)



		
			
	def init_block(self):
		ratio=self.size_cv/self.height
		block_cv=[((int(float(block.x)*ratio),int(float(block.y)*ratio)),(int(float(block.x)*ratio)+int(float(block.w)*ratio),int(float(block.y)*ratio)+int(float(block.h)*ratio))) for block in self.block_list] 
		return block_cv

	def state_open_cv(self):
		ratio=self.size_cv/self.height
		img = np.ones((int(self.size_cv),int(self.size_cv)))
		img = np.array(img * 255, dtype = np.uint8)
		# adding block 
		for block in self.block_cv: 
			(start, end) = block
			color = (150, 150, 150)
			thickness = -1
			img = cv2.rectangle(img, start,end, color, thickness)
		# circle 
		center = (int(float(self.agent.x)*ratio),int(float(self.agent.y)*ratio))
		# radius = int(self.agent.r*ratio)
		color = (0, 0, 0)
		thickness = -1
		img= cv2.circle(img, center, 3, color, thickness)

		# downscaling
		# dim = (self.open_cv_dim,self.open_cv_dim)

		# resize image
		# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

		# expand_dims : 1 channel GRAY SCALE => (1,84,84)
		img=np.expand_dims(img,axis=0)

		# normalise
		img=img/255.0 - 0.5

		# # # Displaying the image 
		# cv2.imshow("image", img) 
		# # # waits for user to press any key
		# # # (this is necessary to avoid Python kernel form crashing)
		# cv2.waitKey(0)

		return img
					

		
	def save(self, filename : str):
		file=filename + str(".json")
		path=os.path.join( os.path.dirname( __file__ ), 'Mazes/' )
		path=path+file

		data={}
		with open(path, 'w') as outfile:
			# Maze data
			maze_data={}
			maze_data["width"]=self.width
			maze_data["height"]=self.height
			maze_data["xinit"]=self.xinit
			maze_data["yinit"]=self.yinit
			maze_data["xgoal"]=self.xgoal
			maze_data["ygoal"]=self.ygoal
			data["maze_data"]=maze_data
			# Blocks data
			blocks_data={}
			for i in range(len(self.block_list)): 
				block_data={}
				block_data["x"] = self.block_list[i].x
				block_data["y"] = self.block_list[i].y
				block_data["w"] = self.block_list[i].w
				block_data["h"] = self.block_list[i].h
				blocks_data["block_"+str(i)]=block_data
			data["blocks_data"]=blocks_data
			json.dump(data, outfile)
			
	def load(self,filename : str):
		file=filename + str(".json")
		path=os.path.join( os.path.dirname( __file__ ), 'Mazes/' )
		path=path+file
		with open(path) as json_file:
			data = json.load(json_file)
		# Maze data
		if "width" in data["maze_data"].keys():
			self.width=data["maze_data"]["width"]
			self.height=data["maze_data"]["height"]
		self.xinit=data["maze_data"]["xinit"]
		self.yinit=data["maze_data"]["yinit"]
		self.xgoal=data["maze_data"]["xgoal"]
		self.ygoal=data["maze_data"]["ygoal"]
		# Blocks data
		# blocks
		for block_data in data["blocks_data"].values(): 
			x=block_data["x"]
			y=block_data["y"]
			w=block_data["w"]
			h=block_data["h"]
			self.add_block(x,y,w,h)
		







# if __name__=="__main__":
	# env =CMaze(filename="easy")
	# print(env.xinit)

	# copied = ray.get(ray.put(env))
	# print(copied.xinit)
	# env=CMaze()
	# env.add_block(10,0,10,90)
	# env.add_block(10,80,80,10)
	# env.add_block(30,60,70,10)
	# env.add_block(10,40,80,10)
	# env.add_block(30,20,70,10)
	# plt.ion()
	# fig, axs = plt.subplots(2, 2)
	# # fig=plt.figure()
	# # gs = fig.add_gridspec(2, 2)
	# # ax1 = fig.add_subplot(gs[0, 0])
	# # ax2 = fig.add_subplot(gs[0, 1])
	# # ax3 = fig.add_subplot(gs[1, :])
	# env.render_matplot(ax=axs[0,0])
	# env.render_matplot(ax=axs[0,0])
	# # show
	# fig.canvas.draw()
	# fig.canvas.flush_events()
		
	# time.sleep(100)

	# state=env.reset()
	# # env.render("C")
	# for k in range(10000):
	# 	action=np.array([-1,1])
	# 	state, reward, done, info = env.step(action)

	# 	env.render()
