#!/bin/bash
#nbo version used: 7.0.10
#orca version used: 5.0.4
export NBOEXE=/path/to/nbo/bin/nbo7.i4.exe
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK #SYSTEM DEPENDENT

module load mpi/openmpi-4.1.5 #remember to load mpi for parallel jobs!

cd /path/to/main/dir/
./orcadir/orca ./orcainputs/orcajob.inp > /path/to/outputdir/orcaoutput.out                                   
