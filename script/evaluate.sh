#!/bin/bash
set -x

# D2E
python src/main.py test \
       --test-file test.jsonl \
       --hierarchical-type d2e \
       --model-path models/d2e.t*/*best* \
       --vocab-file data/word_vocab_test.pickle \
       --output-dir output/d2e_trees.hierarchical

python tools/evaluate_trees.py \
       --tgt-dir output/d2e_trees.hierarchical/ \
       --json-file data/test.jsonl 

# D2E (hard boundary)
python src/main.py test \
       --test-file test.jsonl \
       --hierarchical-type d2e \
       --model-path models/d2e.t*/*best* \
       --vocab-file data/word_vocab_test.pickle \
       --use-hard-boundary \
       --output-dir output/d2e_trees.hierarchical.hard

python tools/evaluate_trees.py \
       --tgt-dir output/d2e_trees.hierarchical.hard/ \
       --json-file data/test.jsonl 

# D2S2E
python src/main.py test \
       --test-file test.jsonl \
       --hierarchical-type d2s2e \
       --model-path models/d2s.t*/*best* models/s2e.t*/*best* \
       --vocab-file data/word_vocab_test.pickle \
       --output-dir output/d2s2e_trees.hierarchical

python tools/evaluate_trees.py \
       --tgt-dir output/d2s2e_trees.hierarchical/ \
       --json-file data/test.jsonl 

# D2P2S2E
python src/main.py test \
       --test-file test.jsonl \
       --hierarchical-type d2p2s2e \
       --model-path models/d2p.t*/*best* models/p2s.t*/*best* models/s2e.t*/*best* \
       --vocab-file data/word_vocab_test.pickle \
       --output-dir output/d2p2s2e_trees.hierarchical

python tools/evaluate_trees.py \
       --tgt-dir output/d2p2s2e_trees.hierarchical/ \
       --json-file data/test.jsonl
