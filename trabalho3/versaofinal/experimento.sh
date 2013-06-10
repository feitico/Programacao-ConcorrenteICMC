#!/bin/bash
for j in {1..5}
do
    for i in {1..18}
    do
        ./run_nodes.sh $i 100 >>  "exp/nodes"${i}"run"${j}".txt"
    done
done
