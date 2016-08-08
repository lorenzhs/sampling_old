#!/bin/bash

universe=$((2**30))

# sequential
for logsamples in {7..27}; do
    echo "==============================================="
    samples=$((2**$logsamples))
    iterations=$((2**30 / $samples))
    echo "Running with 2^$logsamples samples ($iterations iterations)"
    time ./rand32-pgo -t 1 -k $samples -n $universe -i $iterations $@
    time ./rand32S-pgo -t 1 -k $samples -n $universe -i $iterations $@
    echo "==============================================="
    echo ""
done

echo ""
echo "================================================================"
echo "================================================================"
echo "======================= Now with N=2^62 ========================"
echo "================================================================"
echo "================================================================"
echo ""
echo ""
universe=$((2**62))

# sequential
for logsamples in {7..27}; do
    echo "==============================================="
    samples=$((2**$logsamples))
    iterations=$((2**30 / $samples))
    echo "Running with 2^$logsamples samples ($iterations iterations)"
    time ./rand64-pgo -t 1 -k $samples -n $universe -i $iterations $@
    time ./rand64S-pgo -t 1 -k $samples -n $universe -i $iterations $@
    echo "==============================================="
    echo ""
done
