#!/bin/bash

echo "in restart script"

if test "$(kubectl get pods | grep -c 'raltai-gpu')" -gt 0 
then
    if test "$(kubectl get pods | grep -c '1/1')" -eq 0
    then
        kubectl delete pod raltai-gpu
        echo "delete pod raltai-gpu"
    # assumption that container is called c1, but I'm sure no one will notice
    elif test "$(kubectl exec -c c1 raltai-gpu -- nvidia-smi --query-gpu memory.used --format=csv,noheader,nounits)" -eq 0
    then
        kubectl delete pod raltai-gpu
        echo "delete running pod raltai-gpu"
    else
        echo "raltai-gpu still running!"
        exit 1
    fi 
fi 

echo "start new gpu"
bash ~/launch_gpu.sh
exit 0
