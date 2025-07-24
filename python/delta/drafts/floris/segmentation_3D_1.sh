#!/bin/bash
#
#SBATCH --job-name=fc_it
#
#SBATCH --account=ild@gpu           # compte GPU
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=4         # nombre de taches MPI par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4                # nombre de GPU par noeud
#SBATCH --cpus-per-task=10          # nombre de coeurs CPU par tache (un quart du noeud ici)
#
#SBATCH --output=/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/logs/fc_it.log
#SBATCH --error=/gpfsdswork/projects/rech/ild/uqk67mt/spectrif/logs/fc_it.err
#
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
# clean modules
module purge
 
# load tensorflow module
module load tensorflow-gpu/py3/2.4.1

# execute code
# training
srun python segmentation_if_3D_2021.py --host jeanzay --blocks 4 --wide_kernel_starts_at_block -1 --kernel_size 3 --filters 16 --dropout 0.1 --batchnorm 1
srun python segmentation_if_3D_2021.py --host jeanzay --blocks 4 --wide_kernel_starts_at_block 4 --kernel_size 3 --filters 16 --dropout 0.1 --batchnorm 1
srun python segmentation_if_3D_2021.py --host jeanzay --blocks 4 --wide_kernel_starts_at_block 2 --kernel_size 3 --filters 16 --dropout 0.1 --batchnorm 1
srun python segmentation_if_3D_2021.py --host jeanzay --blocks 4 --wide_kernel_starts_at_block 3 --kernel_size 3 --filters 16 --dropout 0.1 --batchnorm 1
srun python segmentation_if_3D_2021.py --host jeanzay --blocks 4 --wide_kernel_starts_at_block 1 --kernel_size 3 --filters 16 --dropout 0.1 --batchnorm 1
