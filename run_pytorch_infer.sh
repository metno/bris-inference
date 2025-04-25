#!/bin/bash

# This script is meant to be executed within
# a singularity container where all the 
# needed packages are available through conda
#
# Example:
#
#   srun singularity exec -B ... run-pytorch.py

# Printing GPU information to terminal once
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
fi
sleep 2

# !Remove this if using an image extended with cotainr or a container from elsewhere.!
# Start conda environment inside the container
#$WITH_CONDA

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

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
# new implementation
export NCCL_P2P_DISABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=1

# old implementation
export NCCL_SOCKET_IFNAME=hsn0,hsn1 #,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3 #COL
export NCCL_IB_GID_INDEX=3

# RCCL specific optimization (new implementation)
export RCCL_NIC=hsn0,hsn1
export RCCL_SOCKET_IFNAME=hsn0,hsn1
export RCCL_BUFFSIZE=16777216 #16mb buff size #8388608
export RCCL_THREADS=8


# Enable RCCL logging for debugging
export RCCL_DEBUG=INFO
export RCCL_DEBUG_SUBSYS=ALL

# Set ROCR_VISIBLE_DEVICES so that each task uses the proper GPU
#export ROCR_VISIBLE_DEVICES=1,2,3,4 #$SLURM_LOCALID


# Set ROCR_VISIBLE_DEVICES so that each task uses the proper GPU
#export ROCR_VISIBLE_DEVICES=1,2,3,4 #$SLURM_LOCALID

# Report affinity to check
echo "Rank $SLURM_PROCID --> $(taskset -p $$); GPU $ROCR_VISIBLE_DEVICES"


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
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID

export HSA_FORCE_FINE_GRAIN_PCIE=1
export HYDRA_FULL_ERROR=1
export AIFS_BASE_SEED=1337420

export PYTHONUSERBASE=$VIRTUAL_ENV
export PATH=$PATH:$VIRTUAL_ENV/bin

bris --config=$1 
