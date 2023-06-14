import json 
# from CMaze import CMaze
import sys
import matplotlib.pyplot as plt 
from env.dm_control.Finger import Finger
import numpy as np
import os
import time


path=sys.argv[1]

env= Finger()

fig = plt.figure(figsize=(10,10))
with open(path) as json_file:
	data = json.load(json_file)

plt.ion()

archive_coord_curiosity=data['archive_coord_curiosity']
print(len(archive_coord_curiosity))
plt.figure()
for i in range(0,len(archive_coord_curiosity),1): 
	archive_coord,fi= archive_coord_curiosity[i]
	plt.scatter(archive_coord[2],archive_coord[3],c=fi+0.2, cmap="Reds")
	fig.canvas.draw()
	fig.canvas.flush_events()
	print(i)
print('END§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§')




	