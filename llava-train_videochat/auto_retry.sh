#!/bin/bash

MAX_RETRIES=100
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "Attempt $(($retry_count + 1))/$MAX_RETRIES"
    
    # Run your training command
    bash scripts/condenser_ver0/ver3.18_vanilla/3.sh
    
    # Check exit code
    if [ $? -eq 0 ]; then
        echo "Training completed successfully"
        exit 0
    else
        echo "Training failed with exit code $?, retrying..."
        retry_count=$((retry_count + 1))
        killall python
        sleep 10  # Wait before retrying
    fi
done

echo "Training failed after $MAX_RETRIES attempts"
exit 1