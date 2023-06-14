import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
import numpy as np 
import jax.numpy as jnp
# from utils.jax_utils import knn_jax, knn_convolution

class ICM(nn.Module): 
	def __init__(self, env, alpha_icm=1e-4, p= 5, m=10, beta=0.2, batch_size=2048, max_buffer=10000, gamma=0.99, dim_enc=8,t_window=2) -> None:
		super(ICM,self).__init__()
		self.observation_space, self.action_space=env.observation_space, env.action_space
		# buffer
		# self.S=[]
		# self.S1=[]
		# self.A=[]
		self.start=0
		# percentage of the trajectory
		self.max_buffer=max_buffer
		self.p=p
		self.m=m
		self.batch_size=batch_size
		self.beta=beta
		self.gamma=gamma
		self.t=t_window
		# loss
		self.loss = torch.nn.MSELoss()
		# self.loss_nr =  torch.nn.MSELoss(reduction='mean',size_average=False, reduce=False)
		### LAYERS ###
		# latent
		self.LR1=nn.Linear(self.observation_space.shape[0],32)
		self.LR2=nn.Linear(32,16)
		self.LR3=nn.Linear(16,dim_enc) 
		# inverse
		self.I1=nn.Linear(2*dim_enc,16)
		self.I2=nn.Linear(16,self.action_space.shape[0])
		# forward
		self.F1=nn.Linear(dim_enc+self.action_space.shape[0],16)
		self.F2=nn.Linear(16,16)
		self.F3=nn.Linear(16,dim_enc) 
		# Optimizer
		self.optimizer=torch.optim.Adam(params=self.parameters(),lr=alpha_icm)

	def feature(self,x):
		z=F.relu(self.LR1(x))
		z=F.relu(self.LR2(z))
		z_s=self.LR3(z)
		return z_s

	def inverse(self,z):
		i=F.relu(self.I1(z))
		i=self.I2(i)
		return i

	def forward(self,z):
		f=F.relu(self.F1(z))
		f=F.sigmoid(self.F2(f))
		f=self.F3(f)
		return f
	
	def sample(self):
		sample_id=np.random.randint(0,self.S1.shape[0],min(self.batch_size,self.S1.shape[0]))
		S1=self.S1[sample_id]
		S=self.S[sample_id]
		A=self.A[sample_id]
		# n_S1,n_S,n_A=zip(*random.sample(list(zip(self.S1,self.S,self.A)),min(self.batch_size,len(self.S))))
		# S1,S,A=np.concatenate(n_S1,axis=0),np.concatenate(n_S,axis=0),np.concatenate(n_A,axis=0)
		# print(n_S1)
		return S1,S,A
	
	def __getstate__(self):
		state_dict = self.state_dict()
		return state_dict

	def __setstate__(self, state_dict):
		self.load_state_dict(state_dict)

	def update(self):
		global_loss=0
		global_li=0
		global_lf=0
		global_lreg=0
		for k in range(self.p):
			S1,S,A=self.sample()
			S,A,S1=torch.tensor(S).float(),torch.tensor(A).float(),torch.tensor(S1).float()

			# ********LF********
			PHI_1 =self.feature(S1)
			PHI=self.feature(S)
			xl=torch.concat([PHI,A],dim=1)
			PHI_1_H=self.forward(xl.float())
			LF=self.loss(PHI_1,PHI_1_H)
			# *******LI*******
			x=torch.concat([PHI,PHI_1],dim=1)
			Apred=self.inverse(x)
			LI=self.loss(A,Apred)
			# ******L*******
			Loss=self.beta*LF+(1-self.beta)*LI
			# Loss=self.beta*LF+(1-self.beta)*LI
			global_loss+=Loss
			global_lf+=LF
			global_li+=LI
			# self.optimizer2.apply_gradients(zip(grads, self.trainable_variables))
			self.optimizer.zero_grad()
			Loss.backward()
			self.optimizer.step()
		return global_loss.detach().numpy(), global_li.detach().numpy(), global_lf.detach().numpy()

	def add_buffer(self, states, actions):
		S1v,Sv,Av=[],[],[]
		S1,S,A=[],[],[]
		step_sample=round(100/self.m)
		i_sample=np.arange(1,len(states),step_sample)
		# i_sample=random.sample(range(1, len(states)), step_sample)
		#  add buffer 
		for i in i_sample : 
			# S1
			S1.append(states[i])
			# S
			S.append(states[i-1])
			# A
			A.append(actions[i-1])
		# # S1
		# self.S1.append(np.array(S1))
		# # S
		# self.S.append(np.array(S))
		# # A
		# self.A.append(np.array(A))
		# S1
		S1v.append(np.array(S1))
		# S
		Sv.append(np.array(S))
		# A
		Av.append(np.array(A))
		# concatenate 
		S1v=np.concatenate(S1v,axis=0)
		Sv=np.concatenate(Sv,axis=0)
		Av=np.concatenate(Av,axis=0)
		if not self.start: 
			self.S1=S1v
			self.S=Sv
			self.A=Av
			self.start=True
		else:
			self.S1=np.concatenate((self.S1,S1v),axis=0) 
			self.S=np.concatenate((self.S,Sv),axis=0) 
			self.A=np.concatenate((self.A,Av),axis=0) 
		if self.S.shape[0]>self.max_buffer:
			delete=self.S.shape[0]-self.max_buffer
			# remove_by_index= random.sample(range(0, self.S.shape[0]), delete)
			# self.S1=np.delete(self.S1,remove_by_index, axis=0)
			# self.S=np.delete(self.S,remove_by_index, axis=0)
			# self.A=np.delete(self.A,remove_by_index, axis=0)
			self.S1=self.S1[delete:]
			self.S=self.S[delete:]
			self.A=self.A[delete:]


			

	def get_curiosity(self,states,actions):
		""" takes as input the trajectory and output the curiosity """
		S,A,S1=np.array(states[:-1]),np.array(actions[:-1]), np.array(states[1:])
		S,A,S1=torch.tensor(S[:-1]).float(),torch.tensor(A[:-1]).float(),torch.tensor(S1[:-1]).float()
		with torch.no_grad():
			# LF
			PHI_1 =self.feature(S1)
			L=list(range(-self.t,self.t+1))
			L.remove(0)
			PHI_1_N=PHI_1.numpy()
			series=[np.expand_dims(np.sqrt(np.sum((np.concatenate((np.repeat(np.expand_dims(PHI_1_N[0],axis=0),-t,axis=0),PHI_1_N[:t]),axis=0) if t<0 else 
				  							np.concatenate((PHI_1_N[t:],np.repeat(np.expand_dims(PHI_1_N[-1],axis=0),t,axis=0)),axis=0)            -PHI_1_N)**2,axis=1)),axis=1) for t in L]
			weights_never_give_up=np.min(np.concatenate(series,axis=-1),axis=-1)
			PHI=self.feature(S)
			xl=torch.concat([PHI,A],dim=1)
			PHI_1_H=self.forward(xl.float())
			sample_weight=self.gamma**np.arange(PHI_1_H.shape[0],0,-1)
			LF=torch.sqrt(torch.sum((PHI_1-PHI_1_H)**2,dim=1))
			LF=np.sum(LF.numpy()*weights_never_give_up*sample_weight,axis=0)
		return LF



if __name__=='__main__':
	from env.GymMaze.CMaze import CMaze
	from utils.cur_individual import Individual
	import matplotlib.pyplot as plt
	env=CMaze(filename='easy',time_horizon=100)
	s=env.reset()
	icm=ICM(env)
	population=[Individual.remote(env) for k in range(5)]
	# # for k in range(5):
	res=ray.get([i.eval.remote(env) for i in population])
	# print(res[0]['S'])
	# print(res[0]['A'])
	# z=icm.feature(torch.tensor(np.array(res[0]['S'])).float())
	c=icm.get_curiosity(res[0]['S'], res[0]['A'])
	# add buffer 
	# for r in res : 
	# 	icm.add_buffer(r['S'],r['A'])
	# # train 
	# # # # Update
	# global_losses=[]
	# global_lfs=[]
	# global_lis=[]

	# for k in range(10):
	# 	global_loss, global_li, global_lf = icm.update()
	# 	global_loss, global_li, global_lf = global_loss.detach().numpy(), global_li.detach().numpy(), global_lf.detach().numpy()

	# 	print(global_loss)
	# 	global_losses.append(global_loss)
	# 	global_lis.append(global_li)
	# 	global_lfs.append(global_lf)

	# # print("Curiosity after : ", cur.get_curiosity(np.array(states),np.array(actions)))
	
	# plt.figure()
	# plt.plot(range(10),global_losses, label='Loss')
	# plt.plot(range(10),global_lis, label='LI')
	# plt.plot(range(10),global_lfs, label='LF')
	# plt.legend()
	# plt.show()	