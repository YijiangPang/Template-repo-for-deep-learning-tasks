#!/bin/bash

### Job name
### Mail type
### Mail user
### report file
### Set a limit on the total run time of the job allocation
#SBATCH --job-name=template
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pangyiji@msu.edu
#SBATCH --output=hpcc_report_%j.SLURMout
#SBATCH --time=30:05:00

### Request that a minimum of minnodes nodes be allocated to this job.
### Request total <number> of tasks
### Request that <ntasks> be invoked on each node
### SBATCH --nodes=1
### SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1

### Require <ncpus> number of processors per task
#SBATCH --cpus-per-task=16

### --mem, --mem-per-cpu, and --mem-per-gpu are mutually exclusive.
### Specify the real memory required per node.
### Minimum memory required per allocated CPU
### SBATCH --mem-per-cpu=1000M
#SBATCH --mem=20G

###total number of GPUs required for the job, valid GPU types are k20, k80 and v100
### GPU on each node
### GPU on each task
### SBATCH --gpus-per-node=v100:4
### SBATCH --gpus-per-task=v100:4
#SBATCH --gpus=v100:4


echo "JobID: $SLURM_JOB_ID"
echo "Running on node: `hostname`"

###check module dependency: module spider CUDA/11.6.0
module load CUDA/11.6.0 cuDNN/8.4.1.50-CUDA-11.6.0
module load GCCcore/10.2.0
module load Conda/3

### init virtual environment if needed
source ~/softwares/anaconda3/etc/profile.d/conda.sh
conda activate p38

### install libs
cd /mnt/home/pangyiji/workspace/Template-repo
###pip install pipreqs
###pipreqs --force
###pip install -r requirements.txt
###cat requirements.txt | xargs -n 1 pip install

### the command to run
srun python main.py
