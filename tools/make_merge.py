import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-json_file', required=True, type=Path)
    parser.add_argument('-merge_dir', required=True, type=Path)
    args = parser.parse_args()

    with open(args.json_file) as in_file:
        for line in in_file:
            data = json.loads(line.strip())
            name = data['doc_id']
            conll_text = conll_format(data)

            merge_file = (args.merge_dir / name).with_suffix('.merge')
            with open(merge_file, 'w') as out_file:
                print(conll_text, file=out_file)


def conll_format(data):
    # sentence_idx word_idx surface lemma pos hoge hoge hoge syntactic_tree edu_idx paragraph_idx
    # 1       6       the     the     DT      det     7       O        (NP (NP (DT the)       4       2
    assert len(data['tokens']) == \
        len(data['word_starts_edu']) == \
        len(data['word_starts_sentence']) == \
        len(data['word_starts_paragraph'])

    edu_idx = 0
    sentence_idx = -1  # sentence idx stert by 0
    paragraph_idx = 0

    conll_dataset = []
    for token_idx, (token, start_edu, start_sentence, start_paragraph) in enumerate(zip(
            data['tokens'],
            data['word_starts_edu'],
            data['word_starts_sentence'],
            data['word_starts_paragraph'])):
        if start_edu:
            edu_idx += 1
        if start_sentence:
            sentence_idx += 1
        if start_paragraph:
            paragraph_idx += 1

        elms = [sentence_idx, token_idx+1, token, token.lower(), 'POS', 'HOGE', 'HOGE', 'HOGE', 'SYN', edu_idx, paragraph_idx]
        line = '\t'.join(list(map(str, elms)))
        conll_dataset.append(line)

    return '\n'.join(conll_dataset)


if __name__ == '__main__':
    main()
