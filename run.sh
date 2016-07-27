#!/bin/bash

universe=$((2**30))

# sequential
for logsamples in {10..27}; do
    echo "==============================================="
    samples=$((2**$logsamples))
    iterations=$((2**30 / $samples))
    echo "Running with 2^$logsamples samples ($iterations iterations)"
    time ./rand32 -t 1 -k $samples -n $universe -i $iterations $@
    time ./rand32S -t 1 -k $samples -n $universe -i $iterations $@
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
for logsamples in {10..27}; do
    echo "==============================================="
    samples=$((2**$logsamples))
    iterations=$((2**30 / $samples))
    echo "Running with 2^$logsamples samples ($iterations iterations)"
    time ./rand64 -t 1 -k $samples -n $universe -i $iterations $@
    time ./rand64S -t 1 -k $samples -n $universe -i $iterations $@
    echo "==============================================="
    echo ""
done
