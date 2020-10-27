import argparse
import json
import pickle
from collections import Counter
from nltk import Tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--ns-vocab', required=True)
    parser.add_argument('--relation-vocab', required=True)
    parser.add_argument('--fmt', choices=['pickle', 'tsv'], default='tsv')
    args = parser.parse_args()

    dataset = load(args.train)
    ns_counter, relation_counter = count(dataset)
    save(ns_counter, args.ns_vocab, args.fmt)
    save(relation_counter, args.relation_vocab, args.fmt)

    print('NS vocab size: {}'.format(len(ns_counter)))
    print('Relation vocab size: {}'.format(len(relation_counter)))

    return 0


def load(fname):
    dataset = []
    with open(fname) as f:
        for line in f:
            dataset.append(json.loads(line))

    return dataset


def count(dataset):
    ns_counter = Counter()
    relation_counter = Counter()

    for data in dataset:
        attach_tree = Tree.fromstring(data['labelled_attachment_tree'])
        labels = [attach_tree[p].label() for p in attach_tree.treepositions()
                  if not isinstance(attach_tree[p], str) and attach_tree[p].height() > 2]
        for label in labels:
            ns, relation = label.split(':')
            ns_counter[ns] += 1
            relation_counter[relation] += 1

    return ns_counter, relation_counter


def save(counter, file_path, fmt):
    if fmt == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(counter, f)
    else:
        with open(file_path, 'w') as f:
            text = '\n'.join(['{}\t{}'.format(k, v) for k, v in counter.most_common()])
            print(text, file=f)

    return


if __name__ == '__main__':
    main()
