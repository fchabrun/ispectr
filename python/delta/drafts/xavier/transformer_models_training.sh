#!/bin/bash
#
#SBATCH --job-name=x_trans1
#
#SBATCH --account=ild@gpu           # compte GPU
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=4         # nombre de taches MPI par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4                # nombre de GPU par noeud
#SBATCH --cpus-per-task=10          # nombre de coeurs CPU par tache (un quart du noeud ici)
#
#SBATCH --output=/gpfsdswork/projects/rech/ild/uqk67mt/delta_trans/logs/x_trans1.log
#SBATCH --error=/gpfsdswork/projects/rech/ild/uqk67mt/delta_trans/logs/x_trans1.err
#
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
# clean modules
module purge
 
# load tensorflow module
module load tensorflow-gpu/py3/2.4.1

# execute code
# training
srun python transformer_models_training.py --path jeanzay