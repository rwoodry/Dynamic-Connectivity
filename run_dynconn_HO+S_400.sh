#!/bin/bash
#$ -S /bin/bash
#$ -j y
#$ -l arch=linux-x64
#$ -cwd
#$ -q chrastil.q,shared.q
#$ -pe openmp 4

umask 0

singularity_image=/mnt/chrastil/lab/snlbin/hopper_python.sif

echo Starting DynConn_HO+S_400 on $HOSTNAME with $NSLOTS cores at `date`
echo Subject: $1

singularity run --cleanenv \
    -B /mnt/chrastil:/mnt/chrastil \
    $singularity_image \
    python /mnt/chrastil/lab/users/rob/projects/DynConn/scripts/dynconn_HO+S_400.py $1

echo Finished at `date`
