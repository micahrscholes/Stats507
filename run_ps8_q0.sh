#!/usr/bin/bash
#
# Author: Micah Scholes
# Updated: November 17, 2021
# 79: -------------------------------------------------------------------------

# slurm options: --------------------------------------------------------------
#SBATCH --job-name=ps8q0
#SBATCH --mail-user=mscholes@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB 
#SBATCH --time=30:00
#SBATCH --account=stats507f21_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/%x-%j-4.log

# application: ----------------------------------------------------------------
n_procs=5

# modules 
module load tensorflow

# the contents of this script
cat run_ps8_q0.sh

# run the script
date

cd /home/jbhender/github/Stats507_F21/demo/
python ps8_q0.py.py $n_procs

date
echo "Done."
