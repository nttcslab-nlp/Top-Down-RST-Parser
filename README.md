# Top-Down RST Parser
This repository is the implementation of "Top-down RST Parsing Utilizing Granularity Levels in Documents" published at AAAI 2020.

## Requirements
python 3.6 or newer  
libraries:
- allennlp==0.9.0
- h5py==2.10.0
- nltk==3.5
- numpy==1.18.4
- torch==1.5.0
- torchtext==0.6.0
- tqdm==4.46.0


## Usage
We use [trees.py](https://github.com/mitchellstern/minimal-span-parser/blob/master/src/trees.py) in our code.
Please put it in `src/dataset/`.

### Preprocess
Before running a script, you need to add a path to Dataset preprocessed by
[Heilman's code](https://github.com/EducationalTestingService/discourse-parsing) into `script/preprocess.sh`.

```bash
bash script/preprocess.sh
```

### Training
Train the model 5 times for D2E, D2P, D2S, P2S, P2E and S2E.
If you need to select a GPU device, please use an environment variable `CUDA_VISIBLE_DEVICES`.

```bash
bash script/training.sh
```

### Evaluating
Evaluate on test set for D2E, D2S2E and D2P2S2E with 5 ensemble setting.

```bash
bash script/evaluate.sh
```

## Data format

We use RSTDT dataset preprocessed [by Heilman's code](https://github.com/EducationalTestingService/discourse-parsing).
In our preprocessing, each data take following jsonl format.
There is sample files of our preprocessing in `data/sample/`.

```bash
"doc_id": "wsj_****"
"rst_tree": "(ROOT (nucleus:Span (text 0) (satellite:Elaboration (text 1))))"
"labelled_attachment_tree": "(nucleus-satellite:Elaboration (text 0) (text 1))"
"tokenized_strings": ["first sentence corresponding to text 1 .", "and this is second sentence ."]
"raw_tokenized_strings": ["first", "sentence", "corresponding", "to", "text", "1", ".", "and", "this", "is", "second", "sentence", "."]
"starts_sentence": [true, true]
"starts_paragraph": [true, false]
"parent_label": null
"granularity_type": D2E
```


## Reference

```
@inproceedings{Kobayashi2020TopDownRP,
  title={Top-Down RST Parsing Utilizing Granularity Levels in Documents},
  author={Naoki Kobayashi and Tsutomu Hirao and Hidetaka Kamigaito and Manabu Okumura and Masaaki Nagata},
  booktitle={Proceedings of the 2020 Conference on Artificial Intelligence for the American (AAAI)},
  month={sep},
  year={2020},
  pages={8099--8106}
}
```

## LICENSE

This software is released under the NTT License, see `LICESCE.txt`.

According to the license, it is not allowed to create pull requests. Please feel free to send issues.
