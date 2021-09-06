#!/bin/bash
#SBATCH -p ipuq                      # partition (queue)
#SBATCH -N 1                         # number of nodes
#SBATCH -n 1                         # number of cores
#SBATCH -t 1-1:00 # time (D-HH:MM)
#SBATCH --exclusive

ulimit -s 10240

module purge
module load slurm/20.02.6
module load graphcore/vipu/1.14.0
module load graphcore/sdk/2.1.0

export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p64_cl_a01_a16.conf
srun ./main
exit 0