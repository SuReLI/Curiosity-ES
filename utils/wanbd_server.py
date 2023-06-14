import wandb
import os
from pathlib import Path
class WandbServer:
	def __init__(self,project_name,name=None) -> None:
		# wandb ENV variables
		self.hostname=os.environ['HOSTNAME']
		if 'armos' in self.hostname:
			os.environ["WANDB_API_KEY"] = "local-29d7f31a540eb70ca381cf9e7e2ea04895912300"
			os.environ["WANDB_BASE_URL"] = "http://armos-lx10.tls.fr.eu.airbus.corp:8100"
			os.environ["WANDB_MODE"] = "dryrun"
			os.environ["WANDB_DIR"] = "../"
		else : 
			os.environ["WANDB_API_KEY"] = "524a47ab6550c03c32c0e9c5df8e6a4c404c2bd5"
			os.environ["WANDB_MODE"] = "dryrun"
			os.environ["WANDB_DIR"] = "../"
		# init
		wandb.init(project=project_name)
		wandb.run.name=name if name !=None else wandb.run.name
	
	