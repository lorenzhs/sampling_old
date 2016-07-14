#!/bin/bash

U='2^31'
size=1000
iterations=100

# sequential
for logsize in {10..30}; do
    echo "==============================="
    echo "Running with 2^$logsize samples"
    size=$((2**$logsize))
    ratio=$(echo "$size / ($U)" | bc -l)
    ./rand -t 1 -s $size -p $ratio -i $iterations $@
    echo "==============================="
    echo ""
done

#for a in {100,1000,10000,100000,1000000}; do
    # generate a*p samples on p processors
#done
