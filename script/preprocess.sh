#!/bash/bin
set -x

# RSTDT=path/to/RSTDT
RSTDT=~/resource/Heilman_tb

DATA=$PWD/data
TOOL=$PWD/tools
mkdir -p $DATA

python tools/preprocess.py \
       -src $RSTDT/rst_discourse_tb_edus_TEST.json \
       -tgt $DATA/test

python tools/preprocess.py \
       -src $RSTDT/rst_discourse_tb_edus_TRAINING_DEV.json \
       -tgt $DATA/valid \
       -divide

python tools/preprocess.py \
       -src $RSTDT/rst_discourse_tb_edus_TRAINING_TRAIN.json \
       -tgt $DATA/train \
       -divide

python tools/label_vocab.py \
       --train $DATA/train.d2e.jsonl \
       --ns-vocab $DATA/ns.vocab \
       --relation-vocab $DATA/relation.vocab

python tools/word_vocab.py \
       --train $DATA/train.d2e.jsonl \
       --valid $DATA/valid.d2e.jsonl \
       --test $DATA/test.jsonl \
       --vocab $DATA/word_vocab_full.pickle

python tools/word_vocab.py \
       --test $DATA/test.jsonl \
       --vocab $DATA/word_vocab_test.pickle

python tools/make_hdf.py \
       --json-file $DATA/train.d2e.jsonl $DATA/valid.d2e.jsonl $DATA/test.jsonl \
       --hdf-file $DATA/vectors.hdf \
       --vocab-file $DATA/word_vocab_full.pickle
       
