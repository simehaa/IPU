#!/bin/bash
#SBATCH -p ipuq                      # partition (queue)
#SBATCH -N 1                         # number of nodes
#SBATCH -n 1                         # number of tasks
#SBATCH -c 32                        # number of cpus per task
#SBATCH -o slurm-32-IPUs-job-%j.out  # STDOUT
#SBATCH -e slurm-32-IPUs-job-%j.err  # STDERR

ulimit -s 10240

module purge
module load slurm/20.02.6
module load graphcore/vipu/1.14.3
module load graphcore/sdk/2.2.0

export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p64_cl_a01_a16.conf
srun ./main --num-ipus 32
exit 0