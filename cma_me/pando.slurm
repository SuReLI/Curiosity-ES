#!/bin/bash

#SBATCH --nodes=2

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=24

#SBATCH --partition=long

#SBATCH --time=4-00:00:00

#SBATCH --begin=now

#SBATCH --job-name=cma_me_main_dm_control_maze

#SBATCH -o cma_me_log.out # STDOUT

#SBATCH -e cma_me_log.err # STDERR

#MODULE LOAD 
module load gcc
conda activate ray

redis_password=$(uuidgen)

export redis_password


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names

nodes_array=($nodes)


node_1=${nodes_array[0]}

ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address


# if we detect a space character in the head node IP, we'll

# convert it to an ipv4 address. This step is optional.

if [[ "$ip" == *" "* ]]; then

  IFS=' ' read -ra ADDR <<< "$ip"

  if [[ ${#ADDR[0]} -gt 16 ]]; then

    ip=${ADDR[1]}

  else

    ip=${ADDR[0]}

  fi

  echo "IPV6 address detected. We split the IPV4 address as $ip"

fi


port=6379

ip_head=$ip:$port

export ip_head

echo "IP Head: $ip_head"


echo "STARTING HEAD at $node_1"

srun --nodes=1 --ntasks=1 -w "$node_1" \

  ray start --head --object-store-memory=$((30000 * 1024 * 1024)) --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &

sleep 30


worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node

for ((i = 1; i <= worker_num; i++)); do

  node_i=${nodes_array[$i]}

  echo "STARTING WORKER $i at $node_i"

  srun --nodes=1 --ntasks=1 -w "$node_i" ray start  --object-store-memory=$((30000 * 1024 * 1024))   --address "$ip_head" --redis-password="$redis_password" --block &

  sleep 5

done

export WANDB_MODE='online'
export XPSLURM='True'

# ===== Call your code below =====

python3 -u main_dm_control.py "$SLURM_CPUS_PER_TASK"
# python3 -u main_maze.py "$SLURM_CPUS_PER_TASK"
