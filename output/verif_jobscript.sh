#!/bin/bash
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_465000527
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=verif
#SBATCH --cpus-per-task=7

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Optional! Set NCCL debug output to check correct use of aws-ofi-rccl (these are very verbose)
#export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,COLL
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3


# Function to find an unused TCP port starting from a specified port number.
find_unused_port() {
    local port=$1
    while : ; do
        # Check if the port is in use
        if ! ss -tuln | grep -qE ":::$port|0.0.0.0:$port" ; then
            # Port is not in use
            echo $port
            break
        fi
        # Increment port by 1 and check again.
        ((port++))
    done
}

# fetches slurm nodelist
get_master_node() {
    # Get the first item in the node list
    first_nodelist=$(echo $SLURM_NODELIST | cut -d',' -f1)

    if [[ "$first_nodelist" == *'['* ]]; then
        # Split the node list and extract the master node
        base_name=$(echo "$first_nodelist" | cut -d'[' -f1)
        range_part=$(echo "$first_nodelist" | cut -d'[' -f2 | cut -d'-' -f1)
        master_node="${base_name}${range_part}"
    else
        # If no range, the first node is the master node
        master_node="$first_nodelist"
    fi

    echo "$master_node"
}

export MASTER_ADDR=$(get_master_node)
export MASTER_PORT=$(find_unused_port 29500)
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

# CXI stuff
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# Enable verbose hydra error outputs in Anemoi
export HYDRA_FULL_ERROR=1

# Set this virtual environment
export VIRTUAL_ENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/.venv

# Ensure the virtual environment is loaded inside the container
export PYTHONUSERBASE=$VIRTUAL_ENV
export PATH=$PATH:$VIRTUAL_ENV/bin

#Should not have to change these
CONFIG_DIR=$(pwd -P)

CONTAINER=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/anemoi_container.sif
VENV=/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/.venv
export VIRTUAL_ENV=$VENV

module load LUMI/23.09 partition/G
# module load LUMI/24.03 partition/C
export SINGULARITYENV_LD_LIBRARY_PATH=/opt/ompi/lib:${EBROOTAWSMINOFIMINRCCL}/lib:/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:${SINGULARITYENV_LD_LIBRARY_PATH}

export PYTHONUSERBASE=$VIRTUAL_ENV
export PATH=$PATH:$VIRTUAL_ENV/bin
OUTPUT_PATH_SAN=/pfs/lustrep4/scratch/project_465000527/multi-domain/inference/output/multidomain_MEPS_sanity_2/verif
OUTPUT_PATH=/pfs/lustrep4/scratch/project_465000527/multi-domain/inference/output/multidomain_MEPS/verif
# OUTPUT_PATH=/pfs/lustrep4/scratch/project_465000527/multi-domain/inference/output/multidomain_AA_sanity_2/verif


srun \
singularity exec -B /pfs:/pfs \
	         -B /var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4 \
                    /pfs/lustrep4/scratch/project_465000527/anemoi/containers/anemoi-core-pytorch-2.3.1-rocm-6.0-python-3.11.sif \
                    verif /pfs/lustrep4/scratch/project_465000527/nipentho/verification/nordic/6h/202206_202305/mslp/cloudy_skies_10d.nc /pfs/lustrep4/scratch/project_465000527/nipentho/verification/nordic/6h/202206_202305/mslp/MEPS_2.5km.nc $OUTPUT_PATH/msl.nc $OUTPUT_PATH_SAN/msl.nc -m bias -f msl_bias_MEPS_maps.png -leg BRIS,MEPS,multidomain,sanity -fs 50,20 -type map  -maptype topo
                    #-o 6,12,18,24,30,36,42,48 -d 20221201:20221231
                    #/pfs/lustrep4/scratch/project_465000527/nipentho/verification/nordic/6h/202206_202305/t2m/cloudy_skies_10d.nc /pfs/lustrep4/scratch/project_465000527/nipentho/verification/nordic/6h/202206_202305/t2m/MEPS_2.5km.nc 
                    # verif $OUTPUT_PATH/2t.nc -m rmse -f 2t_image_AA.png 