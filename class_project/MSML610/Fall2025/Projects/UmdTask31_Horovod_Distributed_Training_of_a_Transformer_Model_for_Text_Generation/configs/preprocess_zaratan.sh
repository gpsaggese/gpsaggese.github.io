#!/bin/bash
#SBATCH --account=msml610-class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=preprocess_bookcorpus
#SBATCH --output=preprocess_bookcorpus.out
#SBATCH --error=preprocess_bookcorpus.err

echo "=== Starting preprocessing on Zaratan compute node ==="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"

# Activate your venv
source /scratch/zt1/project/msml610/user/vikranth/venv/bin/activate

# Move into your project directory (update if needed)
cd /scratch/zt1/project/msml610/user/vikranth/UmdTask31_Horovod_Distributed_Training_of_a_Transformer_Model_for_Text_Generation/notebooks


echo "Running preprocessing notebook as script..."
echo "Python: $(which python)"
echo "--------------------------------------------"

# Run your notebook as a pure Python script
python 00_data_preprocessing.py

echo "--------------------------------------------"
echo "=== Preprocessing finished ==="
echo "Finished at: $(date)"
