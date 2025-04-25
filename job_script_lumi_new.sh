#!/bin/bash
#SBATCH --output=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/bris-inference/logs/full.out
#SBATCH --error=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/bris-inference/logs/full.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --account=project_465000527
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --job-name=
#SBATCH --cpus-per-task=7


#Change this
CONFIG_NAME=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/bris-inference/config/multi_domain_inference_MEPS_n320.yaml


#Should not have to change these
PROJECT_DIR=/pfs/lustrep4/scratch/$SLURM_JOB_ACCOUNT
CONTAINER_SCRIPT=$(pwd -P)/run_pytorch_infer.sh
chmod 770 ${CONTAINER_SCRIPT}
CONFIG_DIR=$(pwd -P)
CONTAINER=$PROJECT_DIR/anemoi/containers/bris-inference-pytorch-2.2.2-rocm-5.6.61-py-3.11.5-v2.sif
VENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/.venv
# VENV=/pfs/lustrep4/scratch/project_465000527/multi-domain/salihiar/deode-md/lumi/.venv
export VIRTUAL_ENV=$VENV

#module load LUMI/23.09 partition/G
module load LUMI/24.03 partition/G
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

# MPI + OpenMP bindings: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"

# run run-pytorch.sh in singularity container like recommended
# in LUMI doc: https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch
srun --cpu-bind=$CPU_BIND \
    singularity exec -B /pfs:/pfs \
                     -B /var/spool/slurmd \
                     -B /opt/cray \
                     -B /usr/lib64 \
                     -B /usr/lib64/libjansson.so.4 \
        $CONTAINER $CONTAINER_SCRIPT $CONFIG_NAME