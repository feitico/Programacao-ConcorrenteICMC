#!/bin/bash
mpirun -np $1 --hostfile nodes.txt ./drunk_speaker palavras.txt $2
