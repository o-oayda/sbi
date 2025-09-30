#!/bin/bash

# Check for required arguments
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 --workers <int> --sims <int>"
    exit 1
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --workers)
        WORKERS="$2"
        shift # past argument
        shift # past value
        ;;
        --sims)
        SIMS="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

# Run Python script with arguments

# Validate that WORKERS and SIMS are integers
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Error: --workers must be an integer."
    exit 1
fi
if ! [[ "$SIMS" =~ ^[0-9]+$ ]]; then
    echo "Error: --sims must be an integer."
    exit 1
fi

python dipolesbi/scripts/quick_dipole.py --workers "$WORKERS" --sims "$SIMS"
