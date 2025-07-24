#!/bin/bash
#
#SBATCH --job-name=swinunet_M0
#
#SBATCH --account=ild@v100          # compte GPU
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1         # nombre de taches MPI par noeud
#SBATCH --gres=gpu:1                # nombre de GPU reserve par noeud
#SBATCH --cpus-per-task=4           # nombre de coeurs CPU par tache
#
#SBATCH --output=/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/logs/swinunet_M0.log
#SBATCH --error=/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/logs/swinunet_M0.err
#
#SBATCH --time=04:00:00             # max heures d'entraînement (max ever = 20)

# clean modules
module purge
 
# load tensorflow module
module load pytorch-gpu/py3/2.1.1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# execute code
srun python /lustre/fswork/projects/rech/ild/uqk67mt/ispectr/scripts/ispectr/python/SPE_IT_model_training.py \
--run_mode full \
--model_name swinunet_M0 \
--num_workers 4
