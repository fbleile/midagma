#!/bin/bash
#SBATCH -D /dss/dsshome1/0C/ge86xim2/midagma
#SBATCH -o slurm_logs/jobfarm.%N.%j.out
#SBATCH -J JobFarm
#SBATCH --mail-user=f.bleile@tum.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --get-user-env
#SBATCH --export=ALL

#SBATCH --clusters=cm4
#SBATCH --partition=cm4_std
#SBATCH --qos=cm4_std
#SBATCH --nodes=2
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00

module load slurm_setup
module load jobfarm

source /dss/dsshome1/0C/ge86xim2/miniconda3/etc/profile.d/conda.sh
conda activate strucmi

export TMPDIR=/tmp
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p /tmp/matplotlib

# Make repo imports work (src.*)
export PYTHONPATH="/dss/dsshome1/0C/ge86xim2/midagma:$PYTHONPATH"

# Use the existing command list
taskdb="cmd"
txt_file="${taskdb}.txt"
echo "$txt_file"

# Ensure log dir + command list file exist
mkdir -p slurm_logs
mkdir -p "$(dirname "$txt_file")"
touch "$txt_file"

# Delete previous jobfarm DB/results
rm -f "${taskdb}.db"
rm -rf "${taskdb}.txt_res"

# Start JobFarm
jobfarm start "$txt_file"

