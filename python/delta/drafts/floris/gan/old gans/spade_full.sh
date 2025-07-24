#!/bin/bash
#
#SBATCH --job-name=spade
#
#SBATCH --account=ild@gpu           # compte GPU
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=4         # nombre de taches MPI par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4                # nombre de GPU par noeud
#SBATCH --cpus-per-task=10          # nombre de coeurs CPU par tache (un quart du noeud ici)
#
#SBATCH --output=/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/logs/spade.log
#SBATCH --error=/gpfsdswork/projects/rech/ild/uqk67mt/delta_gan/logs/spade.err
#
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
# clean modules
module purge
 
# load tensorflow module
module load tensorflow-gpu/py3/2.4.1

# execute code
# training
srun python spade_full_2.py --host jeanzay --generator_blocks 128,64,32 --generator_kernel 4-1,4-1,4-1,4-1 --discriminator_blocks 32,64,128 --discriminator_kernel 4-1,4-1,4-1,4-1 --discriminator_dropout 0.1 --generator_learning_rate 0.00001 --discriminator_learning_rate 0.00001 --discriminator_noise_std 0.01 --discriminator_noise_decayfactor 0.9 --discriminator_noise_decayepochs 10 --pretrain_epochs 100 --pretrain_learning_rate 0.0001