from env.dm_control.Ball_in_cup import Ball_in_cup
from env.dm_control.Stacker import Stacker
from env.dm_control.Finger import Finger
import matplotlib.pyplot as plt 
import numpy as np 

# env=Stacker()
# env=Ball_in_cup()
env=Finger()

map=np.zeros(tuple([ 50 for _ in range(env.b_space)]),dtype=object)

width = 480
height = 480
max_frame = 10000
video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)
plt.ion()
fig,ax=plt.subplots()
for i in range(100):
	state,reward,d,i=env.step(1*np.ones(env.observation_space.shape))
	video = np.hstack([env.env.physics.render(height, width, camera_id=0),
							env.env.physics.render(height, width, camera_id=1)])

	img = ax.imshow(video)
	bcoord=i['bcoord']
	print('bccord : ', bcoord)

	# time.sleep(1)
	# plt.pause(0.01)  # Need min display time > 0.0.
	fig.canvas.draw()
	fig.canvas.flush_events()
