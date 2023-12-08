#!/bin/bash
#SBATCH -A ASC23013       # account name
#SBATCH -J demojob        # job name
#SBATCH -N 2              # total number of nodes requested
#SBATCH -n 8             # total number of tasks requested
#SBATCH -p development       # queue name gpu-a100 or development
#SBATCH -t 00:15:30       # expected maximum runtime (hh:mm:ss)

date

epochs=1000

export NCCL_DEBUG=INFO

if [ -z $WORK ] ; then
  export WORK=$HOME
fi
if [ -z $LOGDIR ] ; then
  export LOGDIR=$WORK/logs
fi
if [ ! -d $LOGDIR ] ; then
  mkdir $LOGDIR
fi
if [ -z $SLURM_JOB_NUM_NODES ] ; then
  nodes=1
  gpus_per_node=5
  processes=$(($nodes*$gpus_per_node))
  hosts=(localhost)
  root_addr=localhost
else
  nodes=$SLURM_JOB_NUM_NODES
  gpus_per_node=$(($SLURM_NTASKS/$nodes))
  processes=$SLURM_NTASKS
  hosts=($(scontrol show hostname ${SLURM_NODELIST}))
  root_addr=${hosts[0]}
  export IBRUN_TASKS_PER_NODE=$gpus_per_node
fi
launcher=$(which ibrun 2> /dev/null)
if [ $? -eq 0 ] ; then
  cmd="ibrun -n $processes"
else
  echo "no launcher!"
fi

random_port=$((1024 + RANDOM % 49152))
rank=$PMI_RANK

echo $rank

cmd="$cmd ./launch.sh"
cmd="$cmd $epochs $gpus_per_node $processes $WORK tcp://${root_addr}:8888"
echo $cmd
$cmd

wait

date

#-e $epochs -r $rank -g $gpus_per_node -n $nproc -w $WORK -a $HOST
