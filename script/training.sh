#!/bin/bash
set -x

for x2y in d2e d2p d2s p2s s2e; do
    for i in `seq 1 5`; do
        python src/main.py train \
               --elmo-embed \
               --gate-embed \
               --parent-label-embed \
               --maximize-metric \
               --train-file train.$x2y.jsonl \
               --valid-file valid.$x2y.jsonl \
               --test-file test.jsonl \
               --serialization-dir models/$x2y.t$i \
               --hierarchical-type $x2y
    done
done
