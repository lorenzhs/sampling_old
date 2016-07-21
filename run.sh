#!/bin/bash

universe=$((2**30))

# sequential
for logsamples in {10..27}; do
    echo "==============================================="
    samples=$((2**$logsamples))
    iterations=$((2**30 / $samples))
    echo "Running with 2^$logsamples samples ($iterations iterations)"
    time ./rand -t 1 -k $samples -n $universe -i $iterations $@
    echo "==============================================="
    echo ""
done
