for node in $(sinfo -p scavenger-gpu -h -N -o '%N'); do
    echo "=== $node ==="
    ssh $node "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" 2>/dev/null
done