from env.GymMaze.CMaze import CMaze
import numpy as np 



env=CMaze(xinit=5, yinit=5, xgoal=10, ygoal=50, time_horizon=1000)

env.reset()

action=np.array([[-10],[-10]])
done=False
while not done : 
    state,reward,done,info=env.step(action)
    print('reward : ',reward)
    print('state : ',state)
    print(done)
    print(env.time_horizon)
    print(env.time)
    env.render()

