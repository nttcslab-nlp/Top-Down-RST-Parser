import argparse
import json
from pathlib import Path
from nltk import Tree
from rsteval import rst_parseval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt-dir', nargs='+', type=Path)
    parser.add_argument('--json-file', type=Path, default='data/test.jsonl')
    args = parser.parse_args()

    doc2gold = load_gold_tree(args.json_file)

    systems = {}
    for tgt_dir in args.tgt_dir:
        systems[tgt_dir.name] = {}
        for tree_file in tgt_dir.iterdir():
            doc_id = tree_file.name.split('.')[0]
            tree = load_tree(tree_file)
            systems[tgt_dir.name][doc_id] = tree

    for name, doc2pred in systems.items():
        pred_trees = [t for n, t in sorted(doc2pred.items(), key=lambda x: x[0])]
        gold_trees = [t for n, t in sorted(doc2gold.items(), key=lambda x: x[0])]
        print(name)
        for eval_type in ['span', 'ns', 'relation', 'full']:
            score = rst_parseval(pred_trees, gold_trees, eval_type=eval_type)
            print(eval_type, score)
    return


def load_tree(file_path):
    tree = None
    with open(file_path) as f:
        # line = f.readline()
        line = ''.join(f.readlines())
        tree = Tree.fromstring(line.strip())

    return tree


def load_gold_tree(json_path):
    trees = {}
    with open(json_path) as f:
        for line in f:
            data = json.loads(line.strip())
            doc_id = data['doc_id']
            if 'labelled_attachment_tree' in data:
                tree = Tree.fromstring(data['labelled_attachment_tree'])
            else:
                tree = Tree.fromstring(data['attach_tree'])
            trees[doc_id] = tree

    return trees


if __name__ == '__main__':
    main()
